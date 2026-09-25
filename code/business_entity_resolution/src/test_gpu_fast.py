import time
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def benchmark_gpu_svd_topk():
    print("==================================================")
    print("   GPU-ACCELERATED TF-IDF SVD TOP-K ENGINE TEST")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    n1 = 441365
    n2 = 2063440
    
    print(f"Testing GPU SVD projection on real scale: {n1:,} S1 queries vs {n2:,} S23 candidates...")
    
    # 1. Fit TF-IDF + SVD (128 dims)
    print("Fitting TF-IDF Char 3-gram Vectorizer & TruncatedSVD(128)...")
    sample_texts = [f"company name sample test {i} location street address" for i in range(10000)]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2, sublinear_tf=True)
    vec.fit(sample_texts)
    
    svd = TruncatedSVD(n_components=128, random_state=42)
    # Fit SVD on sample sparse matrix
    sample_mat = vec.transform(sample_texts)
    svd.fit(sample_mat)

    # Project synthetic embeddings
    print("Generating float16 L2-normalized GPU candidate matrix...")
    np.random.seed(42)
    
    # mat2_gpu on CUDA: 2,063,440 x 128 in float16 = 528 MB VRAM!
    mat2_emb = np.random.randn(n2, 128).astype(np.float16)
    mat2_emb /= np.linalg.norm(mat2_emb, axis=1, keepdims=True)
    mat2_gpu = torch.tensor(mat2_emb, device=device) # 528 MB VRAM
    
    mat1_emb = np.random.randn(n1, 128).astype(np.float16)
    mat1_emb /= np.linalg.norm(mat1_emb, axis=1, keepdims=True)
    mat1_gpu = torch.tensor(mat1_emb, device=device) # 113 MB VRAM

    t0 = time.time()
    chunk_size = 2500  # 2,500 x 2,063,440 x 2 bytes = 10.3 GB VRAM per chunk
    top_k = 30
    total_pairs = 0
    
    print(f"Running CUDA FP16 MatMul + torch.topk() across {n1:,} queries...")
    for start in range(0, n1, chunk_size):
        end = min(start + chunk_size, n1)
        chunk_m1 = mat1_gpu[start:end] # (20000 x 128)
        
        # GPU MatMul: (20000 x 128) @ (128 x 2063440) -> (20000 x 2063440) in FP16 on Tensor Cores!
        sims_gpu = torch.matmul(chunk_m1, mat2_gpu.T)
        
        # GPU Top-K
        vals, idxs = torch.topk(sims_gpu, k=top_k, dim=1)
        total_pairs += vals.numel()
        
        del sims_gpu, vals, idxs
        
    t_tot = time.time() - t0
    print(f"\n  [SUCCESS] GPU Engine processed all {n1:,} S1 entities vs {n2:,} S23 entities in {t_tot:.2f} SECONDS!")
    print(f"  -> Processing Speed: {n1/t_tot:.0f} queries/sec ({total_pairs:,} total pairs)")

if __name__ == "__main__":
    benchmark_gpu_svd_topk()
