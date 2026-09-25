import os
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple

def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parses ground truth DataFrame into map: source1_entity_id -> set of matched_entity_ids.
    """
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1 = row.get('source1_entity_id')
        matches = row.get('matched_entity_ids')
        if pd.isna(s1) or not s1:
            continue
        if pd.isna(matches) or not matches or matches == '':
            gt_map[s1] = set()
            continue
        cand_set = set(m.strip() for m in str(matches).split(',') if m.strip())
        gt_map[s1] = cand_set
    return gt_map

def compute_macro_recall_for_pairs(
    candidate_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    k: int = 50,
    s1_metadata_df: pd.DataFrame = None
) -> Dict[str, any]:
    """
    Computes Macro Recall@K (averaged per S1 entity with true matches).
    Returns overall recall, country breakdown, and cardinality breakdown.
    """
    if candidate_df.empty:
        return {'macro_recall': 0.0, 'total_entities_evaluated': 0}

    # Extract top k candidates per source1 entity fast
    cand_df_k = candidate_df.groupby('source1_entity_id').head(k)
    cand_groups: Dict[str, Set[str]] = cand_df_k.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()

    # Map S1 metadata if provided
    country_map = {}
    if s1_metadata_df is not None and 'country' in s1_metadata_df.columns:
        country_map = s1_metadata_df.set_index('entity_id')['country'].fillna('Unknown').to_dict()

    recalls_all = []
    recalls_by_country: Dict[str, List[float]] = {}
    recalls_by_cardinality: Dict[str, List[float]] = {'singleton': [], 'multi_match': []}

    for s1_id, true_matches in gt_map.items():
        if not true_matches:
            continue  # Skip entities with no ground truth matches

        retrieved_k = set(cand_groups.get(s1_id, []))
        hits = len(retrieved_k.intersection(true_matches))
        recall_i = hits / len(true_matches)
        
        recalls_all.append(recall_i)

        # Country breakdown
        ctry = str(country_map.get(s1_id, 'Unknown')).strip().title()
        recalls_by_country.setdefault(ctry, []).append(recall_i)

        # Cardinality breakdown
        if len(true_matches) == 1:
            recalls_by_cardinality['singleton'].append(recall_i)
        else:
            recalls_by_cardinality['multi_match'].append(recall_i)

    macro_recall = np.mean(recalls_all) if recalls_all else 0.0

    country_summary = {
        ctry: (float(np.mean(vals)), len(vals)) for ctry, vals in recalls_by_country.items()
    }
    cardinality_summary = {
        card: (float(np.mean(vals)), len(vals)) for card, vals in recalls_by_cardinality.items() if vals
    }

    return {
        'macro_recall': float(macro_recall),
        'total_eval_entities': len(recalls_all),
        'country_breakdown': country_summary,
        'cardinality_breakdown': cardinality_summary
    }

def compute_recall_at_k_curve(
    candidate_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    s1_metadata_df: pd.DataFrame = None,
    k_values: List[int] = [10, 20, 30, 40, 50]
) -> Dict[int, Dict[str, any]]:
    """
    Computes macro Recall@K across multiple K values.
    """
    gt_map = parse_ground_truth(gt_df)
    results = {}
    for k in k_values:
        res = compute_macro_recall_for_pairs(candidate_df, gt_map, k=k, s1_metadata_df=s1_metadata_df)
        results[k] = res
    return results

def compute_reduction_ratio(n_candidates: int, n_s1: int, n_s23: int) -> float:
    """
    Reduction Ratio = 1 - (n_candidates / (n_s1 * n_s23))
    """
    total_possible = n_s1 * n_s23
    if total_possible == 0:
        return 1.0
    return 1.0 - (n_candidates / total_possible)
