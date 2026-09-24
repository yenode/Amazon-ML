import sys
import os
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import (
    normalize_dataframe_parallel,
    canonicalize_business_name
)
from feature_engineering import FeatureExtractor

def build_labeled_candidate_pairs(s1_df, s2_df, s3_df, gt_df, n_samples=1000, n_negatives_per_positive=4):
    """
    Builds a labeled dataset of pairs for evaluation.
    Positive pairs: drawn directly from ground_truth.
    Negative pairs: Random candidates that are not in the ground_truth matches.
    """
    # 1. Parse ground truth to list of pairs
    gt_pairs = []
    for _, row in gt_df.iterrows():
        s1 = row.get('source1_entity_id')
        matches = row.get('matched_entity_ids')
        if pd.isna(matches) or matches == '':
            continue
        for m in matches.split(','):
            gt_pairs.append({'source1_entity_id': s1, 'candidate_entity_id': m, 'label': 1})
            
    gt_pairs_df = pd.DataFrame(gt_pairs)
    
    # 2. Filter to only entities present in our sample dataframes
    s1_ids = set(s1_df['entity_id'])
    s23_ids = set(s2_df['entity_id']).union(set(s3_df['entity_id']))
    
    valid_positives = gt_pairs_df[
        gt_pairs_df['source1_entity_id'].isin(s1_ids) & 
        gt_pairs_df['candidate_entity_id'].isin(s23_ids)
    ]
    
    # Sample positives
    if len(valid_positives) > n_samples:
        positives = valid_positives.sample(n=n_samples, random_state=42)
    else:
        positives = valid_positives
        
    # 3. Generate random negatives
    s23_pool = list(s23_ids)
    
    negatives = []
    # Build dictionary of true matches for fast lookup
    true_matches_dict = {}
    for _, row in valid_positives.iterrows():
        s1 = row['source1_entity_id']
        if s1 not in true_matches_dict:
            true_matches_dict[s1] = set()
        true_matches_dict[s1].add(row['candidate_entity_id'])

    np.random.seed(42)
    s1_sample = list(positives['source1_entity_id'].unique())
    if not s1_sample:
        s1_sample = list(s1_ids)[:n_samples]
        
    for s1 in s1_sample:
        true_m = true_matches_dict.get(s1, set())
        # Sample candidates
        cands = np.random.choice(s23_pool, size=n_negatives_per_positive * 2, replace=False)
        added = 0
        for c in cands:
            if c not in true_m:
                negatives.append({'source1_entity_id': s1, 'candidate_entity_id': c, 'label': 0})
                added += 1
                if added == n_negatives_per_positive:
                    break
                    
    neg_df = pd.DataFrame(negatives)
    
    # Balance or combine
    labeled_pairs = pd.concat([positives, neg_df], ignore_index=True)
    return labeled_pairs.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    print("==================================================")
    print("   DIAGNOSTIC PASS: FEATURE VALIDATION & TIMING")
    print("==================================================")

    # 1. Load Data
    print("Loading Validation Split Data (Sampling for speed)...")
    base_dir = "data/student_resource/dataset/val_split"
    s1_raw = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", dtype=str).head(5000)
    s2_raw = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", dtype=str).head(10000)
    s3_raw = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", dtype=str).head(10000)
    gt_raw = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)

    # 2. Normalize
    print("Running Parallel Normalization...")
    t0 = time.time()
    s1_norm = normalize_dataframe_parallel(s1_raw)
    s2_norm = normalize_dataframe_parallel(s2_raw)
    s3_norm = normalize_dataframe_parallel(s3_raw)
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    print(f"  -> Normalized {len(s1_norm) + len(s23_norm)} rows in {time.time()-t0:.2f}s")
    
    # 3. Fit TFIDF and Save
    print("Fitting and Saving TF-IDF Vectorizer state...")
    fe = FeatureExtractor()
    all_names = list(s1_norm['norm_name']) + list(s23_norm['norm_name'])
    fe.fit_global_tfidf(all_names)
    fe.save_global_tfidf("data/student_resource/dataset/val_split/tfidf_state.pkl")
    # Test Loading
    fe = FeatureExtractor()
    fe.load_global_tfidf("data/student_resource/dataset/val_split/tfidf_state.pkl")
    
    # 4. Filter French rows & Test Landmark coverage
    print("\n--- French Landmark Inspection ---")
    test_s1_path = "data/student_resource/dataset/test/test_source1.tsv"
    if os.path.exists(test_s1_path):
        test_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str).head(20000)
        france_rows = test_s1[test_s1.get('country', '').fillna('').str.lower() == 'france']
        if not france_rows.empty:
            fr_norm = normalize_dataframe_parallel(france_rows)
            landmarks_found = fr_norm[fr_norm['has_landmark'] == 1]
            print(f"  -> Found {len(france_rows)} France rows in test sample.")
            print(f"  -> Found landmarks in {len(landmarks_found)} ({len(landmarks_found)/len(france_rows):.2%}) of France rows.")
            if not landmarks_found.empty:
                print("  -> Sample extracted landmarks (French):")
                for l in landmarks_found['landmark_text'].head(3):
                    print(f"     - '{l}'")
        else:
            print("  -> No France rows found in test_s1 sample.")
            
    # 5. Build Labeled Pairs
    print("\n--- Building Labeled Candidate Pairs ---")
    labeled_pairs = build_labeled_candidate_pairs(s1_norm, s2_norm, s3_norm, gt_raw, n_samples=2000, n_negatives_per_positive=4)
    print(f"  -> Total Pairs evaluated: {len(labeled_pairs)} (True Matches: {labeled_pairs['label'].sum()}, Non-Matches: {len(labeled_pairs) - labeled_pairs['label'].sum()})")

    # 6. Extract Features Batch Timings
    print("\n--- Feature Extraction Scaling & Benchmarks ---")
    t0 = time.time()
    feature_matrix = fe.extract_features_batch(labeled_pairs[['source1_entity_id', 'candidate_entity_id']], s1_norm, s23_norm)
    t_feat = time.time() - t0
    pairs_per_sec = len(labeled_pairs) / t_feat
    print(f"  -> Time to extract {len(labeled_pairs)} pair features: {t_feat:.3f} seconds ({pairs_per_sec:.0f} pairs/sec)")

    feature_matrix['label'] = labeled_pairs['label'].values

    # 7. Over-Normalization Deep Dive
    print("\n--- Over-Normalization Audit (Hard Numbers) ---")
    negatives = feature_matrix[feature_matrix['label'] == 0]
    positives = feature_matrix[feature_matrix['label'] == 1]
    
    exact_matches_on_negatives = negatives[negatives['exact_name_match'] == 1.0]
    print(f"  -> Exact name matches on TRUE NON-MATCHES (False Positives risk): {len(exact_matches_on_negatives)} out of {len(negatives)} ({len(exact_matches_on_negatives)/len(negatives):.2%})")
    
    print(f"  -> Non-Match Mean Jaro-Winkler: {negatives['name_jaro_winkler'][negatives['name_jaro_winkler'] >= 0].mean():.3f} (Std: {negatives['name_jaro_winkler'][negatives['name_jaro_winkler'] >= 0].std():.3f})")
    print(f"  -> Non-Match Mean TFIDF Cosine: {negatives['name_char_ngram_cosine'].mean():.3f}")
    
    # 8. Per-Feature Discriminative Power (AUC & Means)
    print("\n--- Per-Feature Discriminative Power (AUC) ---")
    features_to_drop = ['source1_entity_id', 'candidate_entity_id', 'label']
    feature_cols = [c for c in feature_matrix.columns if c not in features_to_drop]
    
    print(f"{'Feature Name':<30} | {'AUC':<7} | {'Mean (Match)':<12} | {'Mean (Non-Match)':<12}")
    print("-" * 75)
    for f in feature_cols:
        
        valid_pos = positives[positives[f] >= -0.5] if not f.startswith('has_') else positives
        valid_neg = negatives[negatives[f] >= -0.5] if not f.startswith('has_') else negatives
        
        mean_pos = valid_pos[f].mean() if not valid_pos.empty else 0
        mean_neg = valid_neg[f].mean() if not valid_neg.empty else 0
        
        # Calculate AUC over strictly valid features and those converted to 0 for missing
        y_true = feature_matrix['label']
        y_scores = feature_matrix[f]
        
        # Some features like candidate_sim_rank are lower-is-better, invert for AUC
        if 'rank' in f or f == 'len_diff_ratio':
            y_scores = -y_scores
            
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5
            
        # Flip AUC < 0.5 if it accidentally perfectly anti-correlates
        if auc < 0.5:
             auc = 1.0 - auc
             
        print(f"{f:<30} | {auc:.3f}   | {mean_pos:<12.3f} | {mean_neg:<12.3f}")

if __name__ == "__main__":
    main()
