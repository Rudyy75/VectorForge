import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("build")
import vectorforge
import os

def generate_data(num_vectors, dim):
    print(f"Generating {num_vectors} vectors of dimension {dim}:")
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors).astype(np.uint64)
    queries = np.random.rand(100, dim).astype(np.float32)
    return vectors, ids, queries

def run_flat_benchmark():
    print("\nRunning FlatIndex Benchmark:")
    dim = 128
    num_vectors = 10000
    vectors, ids, queries = generate_data(num_vectors, dim)
    
    index = vectorforge.FlatIndex(dim, vectorforge.MetricType.L2)
    
    # Add vectors
    start = time.time()
    for i in range(num_vectors):
        index.add(int(ids[i]), vectors[i])
    print(f"Added {num_vectors} vectors in {time.time() - start:.3f}s")
    
    # Search
    k = 10
    start = time.time()
    for q in queries:
        _ = index.search(q, k)
    search_time = time.time() - start
    qps = len(queries) / search_time
    print(f"FlatIndex QPS: {qps:.2f} queries/sec")
    
    # Numpy baseline
    start = time.time()
    for q in queries:
        diffs = vectors - q
        dists = np.sum(diffs**2, axis=1)
        topk = np.argsort(dists)[:k]
    np_time = time.time() - start
    np_qps = len(queries) / np_time
    print(f"NumPy QPS: {np_qps:.2f} queries/sec")
    print(f"Speedup vs NumPy: {qps / np_qps:.2f}x")
    
    return {"FlatIndex": qps, "NumPy": np_qps}

def run_ivf_benchmark():
    print("\nRunning IVFIndex Benchmark:")
    dim = 128
    num_vectors = 100000 # 100k
    nlist = 1024
    vectors, ids, queries = generate_data(num_vectors, dim)
    
    pool = vectorforge.ThreadPool(8)
    index = vectorforge.IVFIndex(dim, nlist, vectorforge.MetricType.L2)
    
    print(f"Training IVFIndex with {nlist} clusters...")
    start = time.time()
    # Train on 20k vectors for speed
    index.train(vectors[:20000], pool)
    print(f"Training completed in {time.time() - start:.3f}s")
    
    print("Adding vectors:")
    start = time.time()
    index.add(vectors, ids, pool)
    print(f"Adding completed in {time.time() - start:.3f}s")
    
    # Ground truth with NumPy
    print("Computing ground truth:")
    gt_ids = []
    k = 10
    for q in queries:
        diffs = vectors - q
        dists = np.sum(diffs**2, axis=1)
        topk = np.argsort(dists)[:k]
        gt_ids.append(ids[topk])
    
    nprobes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    
    for nprobe in nprobes:
        index.set_nprobe(nprobe)
        start = time.time()
        dists, res_ids = index.search(queries, k, pool)
        search_time = time.time() - start
        
        # Calculate recall
        recall_sum = 0
        for i in range(len(queries)):
            hit = len(set(res_ids[i]) & set(gt_ids[i]))
            recall_sum += hit / k
        
        recall = recall_sum / len(queries)
        qps = len(queries) / search_time
        print(f"nprobe={nprobe:<4} | Recall@{k}: {recall:.4f} | QPS: {qps:.2f}")
        results.append({"nprobe": nprobe, "recall": recall, "qps": qps})
        
    df = pd.DataFrame(results)
    
    # Plotting
    os.makedirs("benchmarks", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['recall'], df['qps'], marker='o', linestyle='-', linewidth=2)
    for i, row in df.iterrows():
        plt.annotate(f"nprobe={int(row['nprobe'])}", (row['recall'], row['qps']), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel(f"Recall@{k}")
    plt.ylabel("Queries Per Second (QPS)")
    plt.title("IVFIndex Performance: Recall vs QPS (100k vectors, 128-dim)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmarks/recall_vs_qps.png", dpi=300)
    print("Saved plot to benchmarks/recall_vs_qps.png")
    
    return df

if __name__ == "__main__":
    run_flat_benchmark()
    run_ivf_benchmark()
