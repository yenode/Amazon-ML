import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Set, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import rapidfuzz
from rapidfuzz import fuzz, distance
import pickle

class FeatureExtractor:
    """
    Feature Extraction Engine for Candidate Entity Pairs.
    Integrates C++ rapidfuzz string metrics, global TF-IDF char 3-gram cosine,
    acronym matching, numeric Jaccard, position-aware PIN matching, missingness sentinels,
    and candidate similarity ranking.
    """
    def __init__(self):
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted: bool = False
        self.sentence_model = None

    def fit_global_tfidf(self, all_names: List[str]):
        """
        Fits global TF-IDF character 3-gram vectorizer across the entire entity name corpus.
        Prevents IDF state bugs by maintaining global vocabulary & weights.
        """
        valid_names = [n for n in all_names if isinstance(n, str) and n.strip()]
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            min_df=2,
            sublinear_tf=True
        )
        self.tfidf_vectorizer.fit(valid_names)
        self.is_fitted = True

    def save_global_tfidf(self, filepath: str):
        """Pickle the fitted TF-IDF vectorizer to disk."""
        if not self.is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("Vectorizer is not fitted. Cannot save.")
        with open(filepath, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
    def load_global_tfidf(self, filepath: str):
        """Load the fitted TF-IDF vectorizer from disk."""
        with open(filepath, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        self.is_fitted = True

    def _get_acronym(self, name: str) -> str:
        """Extracts acronym from multi-word entity name (ignoring stop words like 'and', 'of')."""
        if not name or not isinstance(name, str):
            return ""
        stop_words = {'and', 'of', 'for', 'in', 'on', 'the', 'to', 'with', 'by', 'at', 'co', 'de', 'la', 'du'}
        tokens = [t for t in name.split() if len(t) > 1 and not t.startswith('_') and t.lower() not in stop_words]
        if len(tokens) <= 1:
            return ""
        return "".join([t[0] for t in tokens if t[0].isalpha()]).lower()

    def _compute_acronym_match(self, name1: str, name2: str) -> float:
        """Checks if one name is an acronym of the other or if acronyms match."""
        if not name1 or not name2:
            return 0.0
        acr1 = self._get_acronym(name1)
        acr2 = self._get_acronym(name2)
        n1_clean = name1.replace(" ", "").lower()
        n2_clean = name2.replace(" ", "").lower()

        if acr1 and (acr1 == n2_clean or acr1 == acr2):
            return 1.0
        if acr2 and (acr2 == n1_clean or acr1 == acr2):
            return 1.0
        return 0.0

    def compute_pair_features(self, row: pd.Series, tfidf_map: Optional[Dict[str, any]] = None) -> Dict[str, float]:
        """
        Computes the complete feature vector x in R^D for a single candidate pair row.
        Row expected to have fields:
          norm_name_1, clean_name_nosuffix_1, norm_address_1, pin_code_1, num_tokens_1, country_1, entity_id_1, has_landmark_1
          norm_name_2, clean_name_nosuffix_2, norm_address_2, pin_code_2, num_tokens_2, country_2, entity_id_2, has_landmark_2
        """
        n1 = str(row.get('norm_name_1', ''))
        n2 = str(row.get('norm_name_2', ''))
        n1_nosuffix = str(row.get('clean_name_nosuffix_1', ''))
        n2_nosuffix = str(row.get('clean_name_nosuffix_2', ''))

        a1 = str(row.get('norm_address_1', ''))
        a2 = str(row.get('norm_address_2', ''))

        pin1 = row.get('pin_code_1', None)
        pin2 = row.get('pin_code_2', None)

        nums1 = set(row.get('num_tokens_1', [])) if isinstance(row.get('num_tokens_1', []), (list, set)) else set()
        nums2 = set(row.get('num_tokens_2', [])) if isinstance(row.get('num_tokens_2', []), (list, set)) else set()

        c1 = str(row.get('country_1', ''))
        c2 = str(row.get('country_2', ''))

        id1 = str(row.get('entity_id_1', ''))
        id2 = str(row.get('entity_id_2', ''))

        # 1. Name Metrics (rapidfuzz C++)
        has_name1 = bool(n1)
        has_name2 = bool(n2)
        has_name_both = 1.0 if (has_name1 and has_name2) else 0.0

        if has_name_both:
            name_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
            name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
            name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
            exact_match = 1.0 if (n1_nosuffix and n1_nosuffix == n2_nosuffix) else 0.0
            acronym_match = self._compute_acronym_match(n1_nosuffix, n2_nosuffix)
            len_diff_ratio = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
        else:
            name_jw = -1.0
            name_token_sort = -1.0
            name_token_set = -1.0
            exact_match = 0.0
            acronym_match = 0.0
            len_diff_ratio = -1.0

        # 2. Address Metrics
        has_addr1 = bool(a1)
        has_addr2 = bool(a2)
        has_address_both = 1.0 if (has_addr1 and has_addr2) else 0.0

        if has_address_both:
            addr_lev = distance.Levenshtein.normalized_similarity(a1, a2)
            addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        else:
            addr_lev = -1.0
            addr_token_set = -1.0

        # Numeric Jaccard Overlap
        if nums1 or nums2:
            intersection = len(nums1.intersection(nums2))
            union = len(nums1.union(nums2))
            addr_num_jaccard = intersection / union if union > 0 else 0.0
        else:
            addr_num_jaccard = -1.0

        # PIN Code Match
        has_pin1 = bool(pin1)
        has_pin2 = bool(pin2)
        has_pin_both = 1.0 if (has_pin1 and has_pin2) else 0.0

        if has_pin_both:
            pin_code_match = 1.0 if pin1 == pin2 else -1.0
        else:
            pin_code_match = 0.0

        # Context Metrics
        country_match = 1.0 if (c1 and c2 and c1.lower() == c2.lower()) else 0.0
        
        # Source pair type: S1-S2 -> 0, S1-S3 -> 1
        source_pair_type = 1.0 if id2.startswith('S3-') else 0.0

        has_landmark_both = 1.0 if (row.get('has_landmark_1', 0) and row.get('has_landmark_2', 0)) else 0.0

        # Interaction Feature: Both Name AND Address are high simultaneously
        name_score_clean = max(0.0, name_token_set)
        addr_score_clean = max(0.0, addr_token_set)
        name_and_addr_interaction = name_score_clean * addr_score_clean
        name_and_addr_high_flag = 1.0 if (name_score_clean >= 0.7 and addr_score_clean >= 0.7) else 0.0

        return {
            'name_jaro_winkler': float(name_jw),
            'name_token_sort_ratio': float(name_token_sort),
            'name_token_set_ratio': float(name_token_set),
            'exact_name_match': float(exact_match),
            'acronym_match': float(acronym_match),
            'len_diff_ratio': float(len_diff_ratio),
            'addr_levenshtein_ratio': float(addr_lev),
            'addr_token_set_ratio': float(addr_token_set),
            'addr_num_overlap_jaccard': float(addr_num_jaccard),
            'pin_code_match': float(pin_code_match),
            'country_match': float(country_match),
            'source_pair_type': float(source_pair_type),
            'name_and_addr_interaction': float(name_and_addr_interaction),
            'name_and_addr_high_flag': float(name_and_addr_high_flag),
            'has_name_both': float(has_name_both),
            'has_address_both': float(has_address_both),
            'has_pin_both': float(has_pin_both),
            'has_landmark_both': float(has_landmark_both)
        }

    def extract_features_batch(
        self,
        pairs_df: pd.DataFrame,
        s1_norm_df: pd.DataFrame,
        s23_norm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Vectorized/batch feature extractor for candidate pairs DataFrame.
        pairs_df expected columns: ['source1_entity_id', 'candidate_entity_id']
        s1_norm_df and s23_norm_df must be indexed or contain normalized fields.
        """
        if pairs_df.empty:
            return pd.DataFrame()

        # Map entity details into merged DataFrame
        s1_dict = s1_norm_df.set_index('entity_id').to_dict('index')
        s23_dict = s23_norm_df.set_index('entity_id').to_dict('index')

        rows_to_process = []
        for _, pair in pairs_df.iterrows():
            id1 = pair['source1_entity_id']
            id2 = pair['candidate_entity_id']

            rec1 = s1_dict.get(id1, {})
            rec2 = s23_dict.get(id2, {})

            merged_row = {
                'entity_id_1': id1,
                'entity_id_2': id2,
                'norm_name_1': rec1.get('norm_name', ''),
                'clean_name_nosuffix_1': rec1.get('clean_name_nosuffix', ''),
                'norm_address_1': rec1.get('norm_address', ''),
                'pin_code_1': rec1.get('pin_code', None),
                'num_tokens_1': rec1.get('num_tokens', []),
                'country_1': rec1.get('country', ''),
                'has_landmark_1': rec1.get('has_landmark', 0),
                'norm_name_2': rec2.get('norm_name', ''),
                'clean_name_nosuffix_2': rec2.get('clean_name_nosuffix', ''),
                'norm_address_2': rec2.get('norm_address', ''),
                'pin_code_2': rec2.get('pin_code', None),
                'num_tokens_2': rec2.get('num_tokens', []),
                'country_2': rec2.get('country', ''),
                'has_landmark_2': rec2.get('has_landmark', 0),
            }
            rows_to_process.append(merged_row)

        merged_df = pd.DataFrame(rows_to_process)

        # Batch TF-IDF Cosine Similarity calculation (if vectorizer fitted)
        name_char_cosine = np.zeros(len(merged_df))
        if self.is_fitted and self.tfidf_vectorizer is not None:
            names1 = merged_df['norm_name_1'].tolist()
            names2 = merged_df['norm_name_2'].tolist()
            mat1 = self.tfidf_vectorizer.transform(names1)
            mat2 = self.tfidf_vectorizer.transform(names2)
            # Dot product of normalized L2 sparse rows
            dots = mat1.multiply(mat2).sum(axis=1)
            name_char_cosine = np.asarray(dots).flatten()

        # Compute pair features per row
        feature_rows = []
        for idx, row in merged_df.iterrows():
            feat_dict = self.compute_pair_features(row)
            feat_dict['name_char_ngram_cosine'] = float(name_char_cosine[idx])
            feat_dict['source1_entity_id'] = row['entity_id_1']
            feat_dict['candidate_entity_id'] = row['entity_id_2']
            feature_rows.append(feat_dict)

        feat_df = pd.DataFrame(feature_rows)

        # Candidate Similarity Rank feature computation
        # Sort candidates per source1_entity_id by combined score and assign candidate_sim_rank
        feat_df['composite_score'] = (
            0.4 * feat_df['name_token_set_ratio'].clip(lower=0) +
            0.3 * feat_df['name_jaro_winkler'].clip(lower=0) +
            0.3 * feat_df['addr_token_set_ratio'].clip(lower=0)
        )
        feat_df['candidate_sim_rank'] = feat_df.groupby('source1_entity_id')['composite_score'].rank(
            ascending=False, method='min'
        )
        feat_df.drop(columns=['composite_score'], inplace=True)

        return feat_df
