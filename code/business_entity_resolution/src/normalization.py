import unicodedata
import re
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set

# --- Regex Patterns ---

# Legal Suffixes mapping pattern (applies to raw/lowercased text BEFORE character stripping)
# We map to canonical tokens with surrounding underscores so character stripping won't break them.
LEGAL_SUFFIX_MAP = [
    (r'\b(pvt\.?\s*ltd\.?|private\s+limited)\b', ' _pvtltd_ '),
    (r'\b(inc\.?|incorporated)\b', ' _inc_ '),
    (r'\b(corp\.?|corporation)\b', ' _corp_ '),
    (r'\b(llc\.?|l\.l\.c\.?)\b', ' _llc_ '),
    (r'\b(ltd\.?|limited)\b', ' _ltd_ '),
    (r'\b(co\.?|company)\b', ' _co_ '),
    (r'\b(gmbh|g\.m\.b\.h\.)\b', ' _gmbh_ '),
    (r'\b(sa|s\.a\.)\b', ' _sa_ '),
    (r'\b(plc|p\.l\.c\.)\b', ' _plc_ '),
]

LEGAL_TOKENS_SET = {'_pvtltd_', '_inc_', '_corp_', '_llc_', '_ltd_', '_co_', '_gmbh_', '_sa_', '_plc_'}

# Street & Road Abbreviations (applies to address)
STREET_ABBREVIATIONS = [
    (r'\b(rd\.?)\b', 'road'),
    (r'\b(st\.?)\b', 'street'),
    (r'\b(ste\.?)\b', 'suite'),
    (r'\b(flr\.?|fl\.?)\b', 'floor'),
    (r'\b(ave\.?|av\.?)\b', 'avenue'),
    (r'\b(blvd\.?)\b', 'boulevard'),
    (r'\b(dr\.?)\b', 'drive'),
    (r'\b(ln\.?)\b', 'lane'),
    (r'\b(pkwy\.?)\b', 'parkway'),
    (r'\b(hwy\.?)\b', 'highway'),
    (r'\b(sq\.?)\b', 'square'),
    (r'\b(ctr\.?|cntr\.?)\b', 'center'),
    (r'\b(bldg\.?)\b', 'building'),
    (r'\b(apt\.?)\b', 'apartment'),
]

# Landmark relative clause markers (English + French)
LANDMARK_REGEX = re.compile(
    r'\b(near|opp|opposite|behind|beside|next\s+to|près\s+de|pres\s+de|en\s+face\s+de|à\s+côté\s+de|a\s+cote\s+de)\b.*$',
    re.IGNORECASE
)

# Position-aware Postal/PIN code regex:
# Matches 5 or 6 digit numbers at plausible boundaries (end of string, after comma, or space-separated)
PIN_CODE_REGEX = re.compile(r'(?:^|[\s,])(\d{5,6})(?:$|[\s,])')

def normalize_unicode_and_clean(text: str) -> str:
    """NFKD unicode decomposition, ASCII conversion, lowercasing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode NFKD decomposition & ASCII encoding
    text_nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = text_nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return ascii_text.lower().strip()

def canonicalize_business_name(raw_name: str) -> Tuple[str, str]:
    """
    Normalizes business name.
    1. Lowercase + Unicode NFKD.
    2. Canonicalize & -> and.
    3. Canonicalize legal suffixes BEFORE punctuation stripping.
    Returns (norm_name, clean_name_nosuffix).
    """
    clean_text = normalize_unicode_and_clean(raw_name)
    if not clean_text:
        return "", ""
    
    # Replace & with and
    clean_text = re.sub(r'&', ' and ', clean_text)

    # 1. Canonicalize legal suffixes on punctuated string
    norm_name = clean_text
    for pattern, replacement in LEGAL_SUFFIX_MAP:
        norm_name = re.sub(pattern, replacement, norm_name)

    # 2. Strip non-alphanumeric except spaces, hyphens, and underscores (to keep _pvtltd_ intact)
    norm_name = re.sub(r'[^a-z0-9\s\-_]', ' ', norm_name)
    norm_name = re.sub(r'\s+', ' ', norm_name).strip()

    # 3. Create nosuffix version (strip canonical legal suffix tokens and any raw legal words)
    nosuffix = norm_name
    for token in LEGAL_TOKENS_SET:
        nosuffix = nosuffix.replace(token, ' ')
    nosuffix = re.sub(r'\s+', ' ', nosuffix).strip()

    return norm_name, nosuffix

def normalize_address(raw_addr: str) -> Dict[str, any]:
    """
    Normalizes address text.
    - Standardizes street abbreviations.
    - Detects landmark relative clauses (English & French) without stripping address text.
    - Extracts positionally-aware PIN code and numeric tokens.
    Returns dict with keys: norm_address, has_landmark, landmark_text, pin_code, num_tokens.
    """
    clean_text = normalize_unicode_and_clean(raw_addr)
    if not clean_text:
        return {
            "norm_address": "",
            "has_landmark": 0,
            "landmark_text": "",
            "pin_code": None,
            "num_tokens": set()
        }

    # Standardize street abbreviations
    addr_std = clean_text
    for pattern, replacement in STREET_ABBREVIATIONS:
        addr_std = re.sub(pattern, replacement, addr_std)

    # Detect landmark clauses (without removing from main address)
    landmark_match = LANDMARK_REGEX.search(addr_std)
    has_landmark = 1 if landmark_match else 0
    landmark_text = landmark_match.group(0).strip() if landmark_match else ""

    # Position-aware PIN code extraction (5-6 digits)
    pin_match = PIN_CODE_REGEX.search(addr_std)
    pin_code = pin_match.group(1) if pin_match else None

    # Clean punctuation for final norm_address
    norm_address = re.sub(r'[^a-z0-9\s\-]', ' ', addr_std)
    norm_address = re.sub(r'\s+', ' ', norm_address).strip()

    # Extract all numeric tokens
    num_tokens = set(re.findall(r'\b\d+\b', norm_address))

    return {
        "norm_address": norm_address,
        "has_landmark": has_landmark,
        "landmark_text": landmark_text,
        "pin_code": pin_code,
        "num_tokens": num_tokens
    }

def _process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to process a single chunk of records in parallel worker."""
    results = []
    for _, row in chunk_df.iterrows():
        b_name = row.get('business_name', '')
        b_addr = row.get('business_address', '')
        c_code = row.get('country', '')
        
        norm_name, nosuffix_name = canonicalize_business_name(b_name)
        addr_info = normalize_address(b_addr)
        
        results.append({
            'entity_id': row.get('entity_id', ''),
            'raw_business_name': b_name,
            'raw_business_address': b_addr,
            'country': c_code if isinstance(c_code, str) else 'Unknown',
            'norm_name': norm_name,
            'clean_name_nosuffix': nosuffix_name,
            'norm_address': addr_info['norm_address'],
            'has_landmark': addr_info['has_landmark'],
            'landmark_text': addr_info['landmark_text'],
            'pin_code': addr_info['pin_code'],
            'num_tokens': list(addr_info['num_tokens']),
            'has_name': 1 if norm_name else 0,
            'has_address': 1 if addr_info['norm_address'] else 0,
            'has_pin': 1 if addr_info['pin_code'] else 0
        })
    return pd.DataFrame(results)

def normalize_dataframe_parallel(df: pd.DataFrame, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Parallel normalization of DataFrame using multiprocessing.Pool.
    Fast chunked processing bypassing pandas.apply bottleneck.
    """
    if df.empty:
        return pd.DataFrame()

    if n_jobs is None or n_jobs < 1:
        n_jobs = max(1, mp.cpu_count() - 1)

    # If small DataFrame, don't pay process pool overhead
    if len(df) < 5000:
        return _process_chunk(df)

    chunks = np.array_split(df, n_jobs)
    with mp.Pool(processes=n_jobs) as pool:
        processed_chunks = pool.map(_process_chunk, chunks)

    return pd.concat(processed_chunks, ignore_index=True)
