import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_dataframe_parallel
from blocking import BlockingEngine
from blocking_eval import parse_ground_truth, compute_recall_at_k_curve, compute_reduction_ratio, compute_macro_recall_for_pairs

def main():
    print("==================================================")
    print("   PHASE 4: CANDIDATE GENERATION & BLOCKING BENCHMARK")
    print("==================================================")

    # 1. Load Validation Data
    val_dir = "data/student_resource/dataset/val_split"
    print(f"Loading Validation Split Data from {val_dir}...")
    s1_raw = pd.read_csv(os.path.join(val_dir, "train_source1.tsv"), sep="\t", dtype=str, quoting=3)
    s2_raw = pd.read_csv(os.path.join(val_dir, "train_source2.tsv"), sep="\t", dtype=str, quoting=3)
    s3_raw = pd.read_csv(os.path.join(val_dir, "train_source3.tsv"), sep="\t", dtype=str, quoting=3)
    gt_raw = pd.read_csv(os.path.join(val_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, quoting=3)

    n_s1 = len(s1_raw)
    n_s23 = len(s2_raw) + len(s3_raw)
    print(f"  -> Corpus Stats: S1={n_s1:,} entities | S2+S3={n_s23:,} entities | Total Cartesian Pairs={n_s1*n_s23:,}")

    # 2. Parallel Normalization
    print("\nRunning Parallel Normalization...")
    t0 = time.time()
    s1_norm = normalize_dataframe_parallel(s1_raw)
    s2_norm = normalize_dataframe_parallel(s2_raw)
    s3_norm = normalize_dataframe_parallel(s3_raw)
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    t_norm = time.time() - t0
    print(f"  -> Normalization completed in {t_norm:.2f}s ({len(s1_norm)+len(s23_norm):,} total rows)")

    # 3. Candidate Generation (Blocking) Execution
    engine = BlockingEngine()
    
    t_block_start = time.time()
    
    df_exact = engine.block_by_exact_keys(s1_norm, s23_norm)
    df_name = engine.block_by_name_tfidf(s1_norm, s23_norm, top_k=30)
    df_addr = engine.block_by_address_tfidf(s1_norm, s23_norm, top_k=30)

    # Isolated Strategy Recall Benchmarking
    gt_map = parse_ground_truth(gt_raw)
    print("\n--- Strategy Isolated Recall@50 Performance ---")
    for name, df_strat in [("Strategy 3 (Exact Keys)", df_exact), 
                           ("Strategy 1 (Name TF-IDF)", df_name), 
                           ("Strategy 2 (Address TF-IDF)", df_addr)]:
        res = compute_macro_recall_for_pairs(df_strat, gt_map, k=50, s1_metadata_df=s1_norm)
        print(f"  -> {name:<30}: Macro Recall@50 = {res['macro_recall']:.4f} ({len(df_strat):,} pairs generated)")

    # 4. Union and Dedup Candidates
    unioned_df = engine.union_and_dedup([df_exact, df_name, df_addr], max_k=50)
    t_block_total = time.time() - t_block_start
    print(f"\n[BENCHMARK] Total Blocking Pipeline Execution Time: {t_block_total:.2f} seconds")

    # 5. Compute Reduction Ratio
    rr = compute_reduction_ratio(len(unioned_df), n_s1, n_s23)
    print(f"  -> Total Unioned Pairs Generated: {len(unioned_df):,}")
    print(f"  -> Reduction Ratio: {rr:.6%} (Pruned {100.0*rr:.4f}% of search space)")

    # 6. Detailed Recall@K Evaluation Curve
    print("\n==================================================")
    print("   COMBINED UNION MACRO RECALL@K CURVE & BREAKDOWN")
    print("==================================================")
    curve_res = compute_recall_at_k_curve(unioned_df, gt_raw, s1_metadata_df=s1_norm, k_values=[10, 20, 30, 40, 50])

    print(f"{'K':<5} | {'Macro Recall@K':<16} | {'Total Evaluated Entities':<25}")
    print("-" * 52)
    for k, res in curve_res.items():
        print(f"{k:<5} | {res['macro_recall']:<16.4f} | {res['total_eval_entities']:<25}")

    res_50 = curve_res[50]
    print("\n--- Recall@50 Country Breakdown ---")
    for ctry, (rec, count) in res_50['country_breakdown'].items():
        print(f"  - {ctry:<15}: Macro Recall = {rec:.4f} (N={count})")

    print("\n--- Recall@50 Cardinality Breakdown ---")
    for card, (rec, count) in res_50['cardinality_breakdown'].items():
        print(f"  - {card:<15}: Macro Recall = {rec:.4f} (N={count})")

    # 7. Write candidate_pairs.tsv
    out_tsv = os.path.join(val_dir, "candidate_pairs.tsv")
    all_s1_ids = s1_norm['entity_id'].tolist()
    engine.write_candidate_pairs(unioned_df, all_s1_ids, out_tsv)

    # 8. Run Submission Validator
    validator_path = "data/student_resource/validate_submission.py"
    if os.path.exists(validator_path):
        print("\n--- Running Official Submission Format Validation ---")
        import subprocess
        cmd = [sys.executable, validator_path, "--candidate_pairs_path", out_tsv]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("Validator stdout:\n", proc.stdout)
        if proc.stderr:
            print("Validator stderr:\n", proc.stderr)
        if proc.returncode == 0:
            print("  [SUCCESS] candidate_pairs.tsv PASSED OFFICIAL VALIDATOR VERIFICATION!")
        else:
            print("  [WARNING] Validator returned error code:", proc.returncode)

if __name__ == "__main__":
    main()
