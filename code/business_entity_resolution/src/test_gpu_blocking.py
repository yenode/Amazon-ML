import time
import numpy as np
import scipy.sparse as sp
import torch

def benchmark_pytorch_gpu_topk():
    print("Testing PyTorch GPU Top-K Matrix Multiplication Acceleration...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Hardware Device: {device} ({torch.cuda.get_device_name(0)})")

    # Generate synthetic sparse matrices matching real data scale: 400k S1 x 2M S23
    n1 = 50000
    n2 = 200000
    vocab = 20000
    
    print(f"Creating sparse test matrices: ({n1} x {vocab}) @ ({n2} x {vocab}).T ...")
    mat1 = sp.random(n1, vocab, density=0.005, format='csr', dtype=np.float32)
    mat2 = sp.random(n2, vocab, density=0.005, format='csr', dtype=np.float32)

    # Vectorized PyTorch GPU batching
    t0 = time.time()
    top_k = 30
    chunk_size = 5000
    
    # Pre-convert mat2 to CUDA sparse or dense GPU chunks if memory allows
    # Converting mat2.T to PyTorch sparse CSR on GPU
    coo2 = mat2.tocoo()
    indices2 = torch.tensor(np.vstack((coo2.row, coo2.col)), dtype=torch.long)
    values2 = torch.tensor(coo2.data, dtype=torch.float32)
    shape2 = coo2.shape
    
    # GPU Tensor for mat2_T (vocab x n2)
    mat2_t_gpu = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((coo2.col, coo2.row)), dtype=torch.long),
        values2,
        (vocab, n2)
    ).to_sparse_csr().to(device)

    total_pairs = 0
    print("Executing GPU-accelerated Sparse MatMul + torch.topk()...")
    
    for start in range(0, n1, chunk_size):
        end = min(start + chunk_size, n1)
        sub_mat1 = mat1[start:end]
        coo1 = sub_mat1.tocoo()
        
        chunk_gpu = torch.sparse_coo_tensor(
            torch.tensor(np.vstack((coo1.row, coo1.col)), dtype=torch.long),
            torch.tensor(coo1.data, dtype=torch.float32),
            coo1.shape
        ).to_sparse_csr().to(device)

        # PyTorch GPU Sparse MatMul: (5000 x vocab) @ (vocab x 200000) -> (5000 x 200000)
        # Note: torch.matmul supports sparse_csr @ sparse_csr on GPU!
        sim_gpu = torch.matmul(chunk_gpu, mat2_t_gpu)
        
        # GPU Top-K
        # Convert sparse result to dense on GPU per chunk for instant CUDA topk
        sim_dense = sim_gpu.to_dense()
        values, indices = torch.topk(sim_dense, k=top_k, dim=1)
        total_pairs += values.numel()

    t_gpu = time.time() - t0
    print(f"  [SUCCESS] GPU processed {n1} rows x {n2} candidates ({total_pairs:,} pairs) in {t_gpu:.2f} seconds!")
    print(f"  -> Processing Speed: {n1/t_gpu:.0f} S1 queries/sec!")

if __name__ == "__main__":
    benchmark_pytorch_gpu_topk()
