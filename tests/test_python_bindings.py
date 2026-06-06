import os
import sys
import numpy as np

# Add the build directory to the Python path so we can import the module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(build_dir)

import vectorforge

def test_flat_index():
    print("Testing FlatIndex:")
    dim = 64
    num_vectors = 1000
    
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    
    # Create index
    index = vectorforge.FlatIndex(dim, vectorforge.MetricType.L2)
    
    # Add vectors
    for i in range(num_vectors):
        index.add(i, vectors[i])
        
    assert index.size() == num_vectors
    print(f"  Added {index.size()} vectors.")
    
    # Search
    query = vectors[10] # Search for the 10th vector
    results = index.search(query, 5)
    
    assert len(results) == 5
    assert results[0].id == 10 # First result should be the exact match
    print(f"  Search works, Top match: {results[0]}")
    
    # Test persistence
    index.save("test_py_flat.index")
    
    loaded_index = vectorforge.FlatIndex(1, vectorforge.MetricType.Cosine)
    loaded_index.load("test_py_flat.index")
    
    assert loaded_index.size() == num_vectors
    print("  Persistence works.")
    
    os.remove("test_py_flat.index")

def test_ivf_index():
    print("\nTesting IVFIndex:")
    dim = 64
    num_vectors = 5000
    nlist = 10
    
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors).astype(np.uint64)
    
    pool = vectorforge.ThreadPool(4)
    
    index = vectorforge.IVFIndex(dim, nlist, vectorforge.MetricType.L2)
    
    print("  Training")
    index.train(vectors, pool)
    assert index.is_trained()
    
    print("  Adding vectors")
    index.add(vectors, ids, pool)
    assert index.size() == num_vectors
    
    index.set_nprobe(2)
    
    print("  Searching")
    queries = vectors[:5] # Search for first 5 vectors
    dists, res_ids = index.search(queries, 5, pool)
    
    assert res_ids.shape == (5, 5)
    assert dists.shape == (5, 5)
    assert res_ids[0][0] == 0 # First query should match vector 0
    print("  Search works, Numpy arrays successfully returned")

if __name__ == "__main__":
    test_flat_index()
    test_ivf_index()
    print("\nAll Python Bindings tests passed successfully")
