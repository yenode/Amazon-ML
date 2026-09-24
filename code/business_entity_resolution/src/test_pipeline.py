import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import (
    normalize_unicode_and_clean,
    canonicalize_business_name,
    normalize_address,
    normalize_dataframe_parallel
)
from feature_engineering import FeatureExtractor
import tempfile

def test_normalization_unit_cases():
    print("--- 1. Running Normalization Unit Tests ---")
    
    # Unicode & Accent
    res = normalize_unicode_and_clean("Café & Restaurant de Paris, Société Anonyme")
    assert "cafe" in res and "societe" in res, f"Unicode NFKD failed: {res}"
    print("  [PASS] Unicode NFKD & ASCII conversion")

    # Legal Suffix Canonicalization & & replacement
    norm, nosuffix = canonicalize_business_name("Tata Motors Pvt. Ltd. & Co.")
    assert "_pvtltd_" in norm, f"Legal suffix canonicalization failed: {norm}"
    assert "_pvtltd_" not in nosuffix, f"Legal suffix stripping failed: {nosuffix}"
    assert "and" in norm, f"& replacement failed: {norm}"
    print("  [PASS] Legal suffix canonicalization & & replacement")

    # Street Abbreviations
    addr_info = normalize_address("123 Main St., Ste. 400, Rd. 5")
    assert "street" in addr_info["norm_address"], f"Street abbreviation failed: {addr_info}"
    assert "suite" in addr_info["norm_address"], f"Suite abbreviation failed: {addr_info}"
    print("  [PASS] Street abbreviation standardization")

    # Landmark Extraction (English + French)
    addr_en = normalize_address("Opposite SBI Bank, MG Road, Bangalore 560001")
    assert addr_en["has_landmark"] == 1, "English landmark detection failed"
    assert "560001" in addr_en["pin_code"], f"PIN extraction failed: {addr_en}"
    assert "mg road" in addr_en["norm_address"], "Main address should remain intact!"

    addr_fr = normalize_address("15 Rue de Rivoli, près de Musée du Louvre, 75001 Paris")
    assert addr_fr["has_landmark"] == 1, "French landmark detection failed"
    assert addr_fr["pin_code"] == "75001", f"French postal code failed: {addr_fr['pin_code']}"
    print("  [PASS] Landmark extraction & PIN code scoping (English + French)")

def test_feature_engineering_unit_cases():
    print("--- 2. Running Feature Engineering Unit Tests ---")
    
    fe = FeatureExtractor()
    corpus = [
        "industrial credit and investment corporation of india",
        "icici bank",
        "tata motors _pvtltd_",
        "tata steel _pvtltd_",
        "amazon web services _inc_"
    ]
    fe.fit_global_tfidf(corpus)
    assert fe.is_fitted, "TF-IDF global vectorizer fit failed"

    # Acronym match test
    acr_match = fe._compute_acronym_match("Industrial Credit and Investment Corporation of India", "ICICI")
    assert acr_match == 1.0, f"Acronym match failed: {acr_match}"
    print("  [PASS] Acronym match logic")

    # Candidate pairs feature matrix test
    s1_df = pd.DataFrame([
        {
            'entity_id': 'S1-001',
            'business_name': 'Tata Motors Pvt. Ltd.',
            'business_address': '100 MG Road, Bangalore 560001',
            'country': 'India',
            'norm_name': 'tata motors _pvtltd_',
            'clean_name_nosuffix': 'tata motors',
            'norm_address': '100 mg road bangalore 560001',
            'pin_code': '560001',
            'num_tokens': {'100', '560001'},
            'has_landmark': 0
        }
    ])

    s23_df = pd.DataFrame([
        {
            'entity_id': 'S2-001',
            'business_name': 'Tata Motors',
            'business_address': '100 MG Rd, Bangalore 560001',
            'country': 'India',
            'norm_name': 'tata motors',
            'clean_name_nosuffix': 'tata motors',
            'norm_address': '100 mg road bangalore 560001',
            'pin_code': '560001',
            'num_tokens': {'100', '560001'},
            'has_landmark': 0
        },
        {
            'entity_id': 'S2-002',
            'business_name': 'Tata Steel Ltd',
            'business_address': '200 Station Road, Mumbai 400001',
            'country': 'India',
            'norm_name': 'tata steel _ltd_',
            'clean_name_nosuffix': 'tata steel',
            'norm_address': '200 station road mumbai 400001',
            'pin_code': '400001',
            'num_tokens': {'200', '400001'},
            'has_landmark': 0
        }
    ])

    pairs_df = pd.DataFrame([
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-002'}
    ])

    feat_df = fe.extract_features_batch(pairs_df, s1_df, s23_df)
    assert len(feat_df) == 2, f"Expected 2 feature rows, got {len(feat_df)}"
    assert feat_df.loc[0, 'exact_name_match'] == 1.0, "Exact name match failed for S2-001"
    assert feat_df.loc[0, 'pin_code_match'] == 1.0, "PIN match failed"
    assert feat_df.loc[1, 'pin_code_match'] == -1.0, "PIN contradiction failed for S2-002"
    assert feat_df.loc[0, 'candidate_sim_rank'] == 1.0, f"Rank 1 failed: {feat_df.loc[0, 'candidate_sim_rank']}"
    assert feat_df.loc[1, 'candidate_sim_rank'] == 2.0, f"Rank 2 failed: {feat_df.loc[1, 'candidate_sim_rank']}"
    print("  [PASS] Pair feature matrix & candidate similarity ranking")

def test_over_normalization_audit():
    print("--- 3. Running Over-Normalization Precision Audit ---")
    
    # Ensure distinct entities do NOT collapse into exact match or false positive
    pairs_to_check = [
        ("Apple Inc.", "Pineapple Inc."),
        ("Tata Motors Ltd.", "Tata Steel Ltd."),
        ("Starbucks Coffee", "Caribou Coffee"),
        ("First National Bank", "Second National Bank")
    ]

    for n1, n2 in pairs_to_check:
        norm1, nosuffix1 = canonicalize_business_name(n1)
        norm2, nosuffix2 = canonicalize_business_name(n2)
        assert nosuffix1 != nosuffix2, f"Over-normalization bug! '{n1}' and '{n2}' collapsed to '{nosuffix1}'"
    print("  [PASS] Over-normalization audit passed (distinct entities preserved)")

def test_integration_sample_data():
    print("--- 4. Running Integration Test on Real Split Data ---")
    val_s1_path = "data/student_resource/dataset/val_split/train_source1.tsv"
    val_s2_path = "data/student_resource/dataset/val_split/train_source2.tsv"
    val_gt_path = "data/student_resource/dataset/val_split/train_ground_truth.tsv"

    if not os.path.exists(val_s1_path):
        print("  [SKIP] Validation split files not found at expected path.")
        return

    print("  Loading 1,000 sample records from val_split...")
    s1_df = pd.read_csv(val_s1_path, sep="\t", dtype=str).head(1000)
    s2_df = pd.read_csv(val_s2_path, sep="\t", dtype=str).head(2000)
    gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)

    # 1. Parallel Normalization
    t0 = time.time()
    s1_norm = normalize_dataframe_parallel(s1_df)
    s2_norm = normalize_dataframe_parallel(s2_df)
    t_norm = time.time() - t0
    print(f"  [PASS] Parallel normalization of 3,000 records took {t_norm:.3f} seconds")

    # 2. Global TF-IDF Fit
    fe = FeatureExtractor()
    all_names = list(s1_norm['norm_name']) + list(s2_norm['norm_name'])
    fe.fit_global_tfidf(all_names)

    # 3. Create sample candidate pairs
    pairs_list = []
    s2_ids = s2_norm['entity_id'].tolist()
    for s1_id in s1_norm['entity_id'].head(100):
        # Pick 5 candidate IDs
        for cand_id in s2_ids[:5]:
            pairs_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})

    pairs_df = pd.DataFrame(pairs_list)

    # 4. Extract Pair Features Batch
    t0 = time.time()
    feat_df = fe.extract_features_batch(pairs_df, s1_norm, s2_norm)
    t_feat = time.time() - t0
    print(f"  [PASS] Feature matrix calculation for {len(feat_df)} candidate pairs took {t_feat:.3f} seconds")

    # Check for NaNs
    null_counts = feat_df.isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} unhandled NaN values in feature matrix!"
    print(f"  [PASS] Feature matrix verified: shape={feat_df.shape}, unhandled_NaNs=0")

def test_tfidf_serialization_regression():
    print("--- 5. Running TF-IDF Serialization Regression Test ---")
    fe1 = FeatureExtractor()
    corpus = ["apple inc", "pineapple ltd", "tata corporation"]
    fe1.fit_global_tfidf(corpus)
    
    # Transform a test string before save
    mat1 = fe1.tfidf_vectorizer.transform(["apple corporation"]).toarray()
    
    # Save, recreate object, load
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        fe1.save_global_tfidf(tmp_path)
        
        fe2 = FeatureExtractor()
        fe2.load_global_tfidf(tmp_path)
        
        # Transform same test string after load
        mat2 = fe2.tfidf_vectorizer.transform(["apple corporation"]).toarray()
        
        # Assert absolute equality
        if not np.array_equal(mat1, mat2):
            raise ValueError(f"Serialization regression! Vector before: {mat1}, Vector after: {mat2}")
        print("  [PASS] Pickled TF-IDF state perfectly matches training state (no drift/leakage)")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    print("==================================================")
    print("   BUSINESS ENTITY RESOLUTION - TEST & AUDIT SUITE")
    print("==================================================")
    test_normalization_unit_cases()
    test_feature_engineering_unit_cases()
    test_over_normalization_audit()
    test_integration_sample_data()
    test_tfidf_serialization_regression()
    print("==================================================")
    print("   ALL TESTS & AUDITS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
