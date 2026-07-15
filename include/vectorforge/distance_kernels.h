#pragma once

#ifdef VECTORFORGE_HAS_CUDA

#include "vectorforge/types.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace vectorforge {
namespace cuda {

// Hand written CUDA kernel wrapper: Single query against N database vectors
// Returns N distances.
void compute_distances_single_query(
    const float* d_query, 
    const float* d_vectors, 
    float* d_distances, 
    size_t num_vectors, 
    size_t dim, 
    MetricType metric,
    cudaStream_t stream = 0
);

// cuBLAS optimized: Q queries against N database vectors
// Returns QxN distances matrix.
void compute_distances_batch(
    const float* d_queries, 
    const float* d_vectors, 
    float* d_distances, 
    size_t num_queries, 
    size_t num_vectors, 
    size_t dim, 
    MetricType metric,
    cudaStream_t stream = 0
);

}
}

#endif
