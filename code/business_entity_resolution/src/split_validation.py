import pandas as pd
import numpy as np
import os
import gc

def main():
    base_dir = "data/student_resource/dataset/train"
    out_train_dir = "data/student_resource/dataset/train_split"
    out_val_dir = "data/student_resource/dataset/val_split"
    
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)

    print("Loading Source 1 and Ground Truth data...")
    s1_df = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", dtype=str)
    gt_df = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)

    # Calculate match cardinality
    def get_cardinality(x):
        if pd.isna(x) or x == '': return 0
        return len(x.split(','))
        
    gt_df['cardinality'] = gt_df['matched_entity_ids'].apply(get_cardinality)
    
    # Bucket cardinality: 0, 1, 2+ (multi)
    gt_df['cardinality_bucket'] = gt_df['cardinality'].apply(lambda x: min(x, 2))
    
    # Merge country and cardinality bucket for stratification
    s1_with_gt = s1_df.merge(gt_df[['source1_entity_id', 'cardinality_bucket']], left_on='entity_id', right_on='source1_entity_id', how='left')
    
    # Create stratify column
    s1_with_gt['stratify_col'] = s1_with_gt['country'].fillna('Unknown') + "_" + s1_with_gt['cardinality_bucket'].astype(str)
    
    print(f"Stratification distribution:\n{s1_with_gt['stratify_col'].value_counts()}")

    # Stratified split using pandas groupby and sample
    np.random.seed(42)
    
    # By grouping and sampling 80% from each strata we preserve the distribution exactly
    train_s1_with_gt = s1_with_gt.groupby('stratify_col', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))
    val_s1_with_gt = s1_with_gt.drop(train_s1_with_gt.index)
    
    # Shuffle the resulting sets
    train_s1_with_gt = train_s1_with_gt.sample(frac=1.0, random_state=42)
    val_s1_with_gt = val_s1_with_gt.sample(frac=1.0, random_state=42)

    # Keep only original columns
    s1_cols = s1_df.columns
    train_s1_out = train_s1_with_gt[s1_cols]
    val_s1_out = val_s1_with_gt[s1_cols]

    print(f"Train Source 1 size: {len(train_s1_out)}")
    print(f"Val Source 1 size: {len(val_s1_out)}")

    # Extract corresponding ground truth
    gt_df.set_index("source1_entity_id", inplace=True)
    
    train_gt = train_s1_out[["entity_id"]].join(gt_df, on="entity_id", how="left")
    val_gt = val_s1_out[["entity_id"]].join(gt_df, on="entity_id", how="left")
    
    train_gt.rename(columns={"entity_id": "source1_entity_id"}, inplace=True)
    val_gt.rename(columns={"entity_id": "source1_entity_id"}, inplace=True)

    train_gt_out = train_gt[["source1_entity_id", "matched_entity_ids"]]
    val_gt_out = val_gt[["source1_entity_id", "matched_entity_ids"]]

    # Save S1 and GT
    print("Saving Source 1 and Ground Truth splits...")
    train_s1_out.to_csv(os.path.join(out_train_dir, "train_source1.tsv"), sep="\t", index=False)
    train_gt_out.to_csv(os.path.join(out_train_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    
    val_s1_out.to_csv(os.path.join(out_val_dir, "train_source1.tsv"), sep="\t", index=False)
    val_gt_out.to_csv(os.path.join(out_val_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    # Partition Source 2 and Source 3 Space
    print("Extracting true matched IDs for S2 and S3...")
    def get_all_matched_ids(gt_dataframe):
        matches = gt_dataframe['matched_entity_ids'].dropna()
        matches = matches[matches != '']
        all_ids = set()
        for id_list in matches:
            for entity in id_list.split(','):
                all_ids.add(entity.strip())
        return all_ids

    train_true_ids = get_all_matched_ids(train_gt_out)
    val_true_ids = get_all_matched_ids(val_gt_out)

    print(f"Train true S2/S3 matches: {len(train_true_ids)}")
    print(f"Val true S2/S3 matches: {len(val_true_ids)}")

    def partition_source_file(filename):
        print(f"Processing {filename}...")
        df = pd.read_csv(os.path.join(base_dir, filename), sep="\t", dtype=str)
        
        is_train_true = df['entity_id'].isin(train_true_ids)
        is_val_true = df['entity_id'].isin(val_true_ids)
        
        is_distractor = ~(is_train_true | is_val_true)
        distractors = df[is_distractor]
        
        train_dist = distractors.sample(frac=0.8, random_state=42)
        val_dist = distractors.drop(train_dist.index)
        
        train_df = pd.concat([df[is_train_true], train_dist]).drop_duplicates(subset=['entity_id'])
        val_df = pd.concat([df[is_val_true], val_dist]).drop_duplicates(subset=['entity_id'])
        
        print(f"  -> Train {filename} size: {len(train_df)}")
        print(f"  -> Val {filename} size: {len(val_df)}")
        
        train_df.to_csv(os.path.join(out_train_dir, filename), sep="\t", index=False)
        val_df.to_csv(os.path.join(out_val_dir, filename), sep="\t", index=False)
        
        del df, train_df, val_df, train_dist, val_dist
        gc.collect()

    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        train_file = os.path.join(out_train_dir, s_file)
        val_file = os.path.join(out_val_dir, s_file)
        if os.path.islink(train_file): os.remove(train_file)
        if os.path.islink(val_file): os.remove(val_file)
        if os.path.exists(train_file): os.remove(train_file)
        if os.path.exists(val_file): os.remove(val_file)
        
        partition_source_file(s_file)

    print("Data successfully partitioned and stratified!")

if __name__ == "__main__":
    main()
