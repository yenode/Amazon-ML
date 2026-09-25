import time
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer

def _process_chunk_vectorized(args):
    """
    Worker function: processes a chunk of S1 rows against S23 sparse matrix.
    Uses direct C-level scipy.csr arrays and numpy argpartition for high-speed top-K.
    """
    chunk_m1, mat2_t, s1_ids_chunk, s23_ids, top_k, min_sim = args
    
    # Fast C-level sparse matrix multiplication: (N_chunk x V) @ (V x N2) -> (N_chunk x N2)
    sim_mat = chunk_m1.dot(mat2_t)  # csr_matrix
    
    indptr = sim_mat.indptr
    indices = sim_mat.indices
    data = sim_mat.data
    
    out_s1 = []
    out_cand = []
    out_scores = []
    
    for i in range(len(s1_ids_chunk)):
        start = indptr[i]
        end = indptr[i+1]
        if start == end:
            continue
            
        row_indices = indices[start:end]
        row_data = data[start:end]
        
        # Filter by min_sim
        mask = row_data >= min_sim
        if not np.any(mask):
            continue
            
        row_indices = row_indices[mask]
        row_data = row_data[mask]
        
        if len(row_data) > top_k:
            top_k_idx = np.argpartition(row_data, -top_k)[-top_k:]
            top_k_idx = top_k_idx[np.argsort(-row_data[top_k_idx])]
            best_cand_idx = row_indices[top_k_idx]
            best_scores = row_data[top_k_idx]
        else:
            sort_idx = np.argsort(-row_data)
            best_cand_idx = row_indices[sort_idx]
            best_scores = row_data[sort_idx]
            
        s1_id = s1_ids_chunk[i]
        for c_idx, sc in zip(best_cand_idx, best_scores):
            out_s1.append(s1_id)
            out_cand.append(s23_ids[c_idx])
            out_scores.append(float(sc))
            
    return out_s1, out_cand, out_scores

def main():
    print("Testing Multi-Core Parallel Vectorized Scipy/NumPy Sparse Top-K...")
    n1 = 50000
    n2 = 200000
    
    print(f"Generating realistic text data ({n1} S1 x {n2} S23)...")
    s1_texts = [f"company name test {i} street location" for i in range(n1)]
    s23_texts = [f"company name test {j % 5000} street location" for j in range(n2)]
    
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
    vec.fit(s1_texts[:1000] + s23_texts[:1000])
    
    print("Transforming TF-IDF matrices...")
    mat1 = vec.transform(s1_texts)
    mat2 = vec.transform(s23_texts)
    mat2_t = mat2.T.tocsc()
    
    s1_ids = np.array([f"S1-{i}" for i in range(n1)])
    s23_ids = np.array([f"S23-{j}" for j in range(n2)])
    
    t0 = time.time()
    n_workers = min(16, mp.cpu_count())
    chunk_size = (n1 + n_workers - 1) // n_workers
    
    tasks = []
    for w in range(n_workers):
        st = w * chunk_size
        en = min(st + chunk_size, n1)
        if st >= en:
            break
        tasks.append((
            mat1[st:en],
            mat2_t,
            s1_ids[st:en],
            s23_ids,
            30,
            0.15
        ))
        
    print(f"Dispatching {len(tasks)} parallel tasks across {n_workers} CPU worker cores...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_chunk_vectorized, tasks)
        
    total_pairs = sum(len(res[0]) for res in results)
    t_tot = time.time() - t0
    
    print(f"\n  [SUCCESS] Parallel Scipy/NumPy engine processed {n1:,} S1 queries vs {n2:,} candidates in {t_tot:.2f} seconds!")
    print(f"  -> Processing Speed: {n1/t_tot:.0f} S1 queries/sec ({total_pairs:,} total pairs generated)!")

if __name__ == "__main__":
    main()
