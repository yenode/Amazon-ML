import sys
import os
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import (
    normalize_dataframe_parallel,
    canonicalize_business_name
)
from feature_engineering import FeatureExtractor

def build_hard_labeled_pairs(s1_norm, s23_norm, gt_df, fe_fitted, n_s1=2000, k=25):
    """
    Builds a labeled dataset using HARD negatives based on actual TF-IDF similarity.
    Retrieves the top K closest candidates for each S1 entity. 
    Labels TRUE matches = 1, TRUE non-matches (but highly text-similar) = 0.
    """
    # Create mapping of true matches
    gt_pairs = []
    for _, row in gt_df.iterrows():
        s1 = row.get('source1_entity_id')
        matches = row.get('matched_entity_ids')
        if pd.isna(matches) or matches == '':
            continue
        for m in matches.split(','):
            gt_pairs.append((s1, m))
    true_matches_set = set(gt_pairs)

    # Subset S1
    if len(s1_norm) > n_s1:
        s1_sample = s1_norm.sample(n=n_s1, random_state=42).copy()
    else:
        s1_sample = s1_norm.copy()

    # TF-IDF vectors
    s1_names = s1_sample['norm_name'].tolist()
    s23_names = s23_norm['norm_name'].tolist()

    mat_s1 = fe_fitted.tfidf_vectorizer.transform(s1_names)
    mat_s23 = fe_fitted.tfidf_vectorizer.transform(s23_names)

    # Nearest neighbors (cosine sim)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(mat_s23)
    
    distances, indices = nn.kneighbors(mat_s1)

    s1_ids = s1_sample['entity_id'].values
    s23_ids = s23_norm['entity_id'].values

    pairs = []
    for i in range(len(s1_ids)):
        s1_val = s1_ids[i]
        for j in range(k):
            s23_val = s23_ids[indices[i, j]]
            label = 1 if (s1_val, s23_val) in true_matches_set else 0
            pairs.append({'source1_entity_id': s1_val, 'candidate_entity_id': s23_val, 'label': label})

    pairs_df = pd.DataFrame(pairs)
    # Filter out entities not technically in the true matches subset to preserve accuracy of label matching
    # i.e., we must only include positive pairs if S1 actually had some ground truth in our current eval.
    
    return pairs_df

def main():
    print("==================================================")
    print("   DIAGNOSTIC PASS 2: HARD NEGATIVES & REAL SCALE")
    print("==================================================")

    # 1. Load Data
    base_dir = "data/student_resource/dataset/val_split"
    # Loading MORE data to prove scale
    s1_raw = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", dtype=str).head(5000)
    s2_raw = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", dtype=str).head(20000)
    s3_raw = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", dtype=str).head(20000)
    gt_raw = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)

    print("Running Parallel Normalization (45k rows)...")
    t0 = time.time()
    s1_norm = normalize_dataframe_parallel(s1_raw)
    s2_norm = normalize_dataframe_parallel(s2_raw)
    s3_norm = normalize_dataframe_parallel(s3_raw)
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    print(f"  -> Normalized 45k rows in {time.time()-t0:.2f}s")
    
    fe = FeatureExtractor()
    all_names = list(s1_norm['norm_name']) + list(s23_norm['norm_name'])
    fe.fit_global_tfidf(all_names)

    # 2. Build HARD labeled pairs
    print("\n--- Building HARD Labeled Candidate Pairs ---")
    print("  -> Retrieving Top K=25 Hard Candidates using TF-IDF Cosine NN...")
    labeled_pairs = build_hard_labeled_pairs(s1_norm, s23_norm, gt_raw, fe, n_s1=2000, k=25)
    print(f"  -> Total Pairs evaluated: {len(labeled_pairs)} (True Matches: {labeled_pairs['label'].sum()}, Hard Non-Matches: {len(labeled_pairs) - labeled_pairs['label'].sum()})")

    if labeled_pairs['label'].sum() == 0:
        print("  -> WARNING: 0 true matches captured. Exiting to avoid computation errors.")
        return

    # 3. Scale Test (100k scale)
    print("\n--- Feature Extraction Scaling & Benchmarks ---")
    scale_pairs = pd.concat([labeled_pairs] * 2, ignore_index=True) # approx 100k pairs (2 * 50k)
    print(f"  -> Benchmarking on {len(scale_pairs)} candidate pairs to unmask overhead.")
    
    t0 = time.time()
    _ = fe.extract_features_batch(scale_pairs[['source1_entity_id', 'candidate_entity_id']], s1_norm, s23_norm)
    t_feat_scale = time.time() - t0
    scale_pairs_per_sec = len(scale_pairs) / t_feat_scale
    print(f"  -> Time to extract {len(scale_pairs)} pair features: {t_feat_scale:.3f} seconds ({scale_pairs_per_sec:.0f} pairs/sec)")
    # 1.2M Extrapolation
    projected_1m2 = (1_200_000 / scale_pairs_per_sec) / 60
    print(f"  -> Projected time for 1.2M pairs (K=50 for ~24k test entities): {projected_1m2:.2f} minutes")

    # 4. Feature Extraction on base pairs
    feature_matrix = fe.extract_features_batch(labeled_pairs[['source1_entity_id', 'candidate_entity_id']], s1_norm, s23_norm)
    feature_matrix['label'] = labeled_pairs['label'].values

    # 5. Discriminative Power (AUC)
    print("\n--- Per-Feature Discriminative Power (HARD AUC) ---")
    features_to_drop = ['source1_entity_id', 'candidate_entity_id', 'label']
    feature_cols = [c for c in feature_matrix.columns if c not in features_to_drop]
    
    positives = feature_matrix[feature_matrix['label'] == 1]
    negatives = feature_matrix[feature_matrix['label'] == 0]
    
    print(f"{'Feature Name':<30} | {'AUC':<7} | {'Mean (Match)':<12} | {'Mean (Hard Neg)':<12}")
    print("-" * 75)
    for f in feature_cols:
        
        valid_pos = positives[positives[f] >= -0.5] if not f.startswith('has_') else positives
        valid_neg = negatives[negatives[f] >= -0.5] if not f.startswith('has_') else negatives
        
        mean_pos = valid_pos[f].mean() if not valid_pos.empty else 0
        mean_neg = valid_neg[f].mean() if not valid_neg.empty else 0
        
        y_true = feature_matrix['label']
        y_scores = feature_matrix[f]
        
        if 'rank' in f or f == 'len_diff_ratio':
            y_scores = -y_scores
            
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5
            
        if auc < 0.5:
             auc = 1.0 - auc
             
        print(f"{f:<30} | {auc:.3f}   | {mean_pos:<12.3f} | {mean_neg:<12.3f}")

if __name__ == "__main__":
    main()
