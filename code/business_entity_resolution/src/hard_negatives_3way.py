import sys
import os
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_dataframe_parallel
from feature_engineering import FeatureExtractor

def build_mined_candidate_pairs(s1_norm, s23_norm, gt_df, mode='name', n_s1=2000, k=25):
    """
    Mines candidate pairs under 3 regimes:
      mode='name': Top-K by Name TF-IDF similarity
      mode='address': Top-K by Address TF-IDF similarity
      mode='blended': Top-K by Blended Name+Address TF-IDF similarity
    """
    gt_pairs = []
    for _, row in gt_df.iterrows():
        s1 = row.get('source1_entity_id')
        matches = row.get('matched_entity_ids')
        if pd.isna(matches) or matches == '':
            continue
        for m in matches.split(','):
            gt_pairs.append((s1, m))
    true_matches_set = set(gt_pairs)

    if len(s1_norm) > n_s1:
        s1_sample = s1_norm.sample(n=n_s1, random_state=42).copy()
    else:
        s1_sample = s1_norm.copy()

    s1_ids = s1_sample['entity_id'].values
    s23_ids = s23_norm['entity_id'].values

    if mode == 'name':
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
        vec.fit(list(s1_sample['norm_name']) + list(s23_norm['norm_name']))
        mat_s1 = vec.transform(s1_sample['norm_name'])
        mat_s23 = vec.transform(s23_norm['norm_name'])
    elif mode == 'address':
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
        vec.fit(list(s1_sample['norm_address']) + list(s23_norm['norm_address']))
        mat_s1 = vec.transform(s1_sample['norm_address'])
        mat_s23 = vec.transform(s23_norm['norm_address'])
    elif mode == 'blended':
        vec_name = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
        vec_name.fit(list(s1_sample['norm_name']) + list(s23_norm['norm_name']))
        vec_addr = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
        vec_addr.fit(list(s1_sample['norm_address']) + list(s23_norm['norm_address']))
        
        m_s1_n = vec_name.transform(s1_sample['norm_name'])
        m_s23_n = vec_name.transform(s23_norm['norm_name'])
        m_s1_a = vec_addr.transform(s1_sample['norm_address'])
        m_s23_a = vec_addr.transform(s23_norm['norm_address'])
        
        from scipy.sparse import hstack
        mat_s1 = hstack([m_s1_n, m_s1_a])
        mat_s23 = hstack([m_s23_n, m_s23_a])

    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(mat_s23)
    distances, indices = nn.kneighbors(mat_s1)

    pairs = []
    for i in range(len(s1_ids)):
        s1_val = s1_ids[i]
        for j in range(k):
            s23_val = s23_ids[indices[i, j]]
            label = 1 if (s1_val, s23_val) in true_matches_set else 0
            pairs.append({'source1_entity_id': s1_val, 'candidate_entity_id': s23_val, 'label': label})

    return pd.DataFrame(pairs)

def run_evaluation(mode, s1_norm, s23_norm, gt_raw, fe):
    print(f"\n==================================================")
    print(f"   EVALUATING REGIME: {mode.upper()} MINED NEGATIVES")
    print(f"==================================================")
    
    pairs_df = build_mined_candidate_pairs(s1_norm, s23_norm, gt_raw, mode=mode, n_s1=2000, k=25)
    print(f"Total Pairs: {len(pairs_df)} (Positives: {pairs_df['label'].sum()}, Hard Negatives: {len(pairs_df) - pairs_df['label'].sum()})")
    
    feat_df = fe.extract_features_batch(pairs_df[['source1_entity_id', 'candidate_entity_id']], s1_norm, s23_norm)
    feat_df['label'] = pairs_df['label'].values
    
    positives = feat_df[feat_df['label'] == 1]
    negatives = feat_df[feat_df['label'] == 0]
    
    features_to_drop = ['source1_entity_id', 'candidate_entity_id', 'label']
    feature_cols = [c for c in feat_df.columns if c not in features_to_drop]
    
    print(f"{'Feature Name':<30} | {'AUC':<7} | {'Mean (Match)':<12} | {'Mean (Hard Neg)':<12}")
    print("-" * 75)
    results = {}
    for f in feature_cols:
        valid_pos = positives[positives[f] >= -0.5] if not f.startswith('has_') else positives
        valid_neg = negatives[negatives[f] >= -0.5] if not f.startswith('has_') else negatives
        
        mean_pos = valid_pos[f].mean() if not valid_pos.empty else 0
        mean_neg = valid_neg[f].mean() if not valid_neg.empty else 0
        
        y_true = feat_df['label']
        y_scores = feat_df[f]
        
        if 'rank' in f or f == 'len_diff_ratio':
            y_scores = -y_scores
            
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5
            
        if auc < 0.5:
             auc = 1.0 - auc
             
        results[f] = (auc, mean_pos, mean_neg)
        print(f"{f:<30} | {auc:.3f}   | {mean_pos:<12.3f} | {mean_neg:<12.3f}")
    return results

def main():
    base_dir = "data/student_resource/dataset/val_split"
    s1_raw = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", dtype=str).head(5000)
    s2_raw = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", dtype=str).head(20000)
    s3_raw = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", dtype=str).head(20000)
    gt_raw = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)

    print("Normalizing 45k records...")
    s1_norm = normalize_dataframe_parallel(s1_raw)
    s2_norm = normalize_dataframe_parallel(s2_raw)
    s3_norm = normalize_dataframe_parallel(s3_raw)
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    
    fe = FeatureExtractor()
    all_names = list(s1_norm['norm_name']) + list(s23_norm['norm_name'])
    fe.fit_global_tfidf(all_names)

    res_name = run_evaluation('name', s1_norm, s23_norm, gt_raw, fe)
    res_addr = run_evaluation('address', s1_norm, s23_norm, gt_raw, fe)
    res_blend = run_evaluation('blended', s1_norm, s23_norm, gt_raw, fe)

if __name__ == "__main__":
    main()
