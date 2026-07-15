#pragma once

#ifdef VECTORFORGE_HAS_CUDA

#include <cuda_runtime.h>
#include "vectorforge/types.h"
#include <cstddef>

namespace vectorforge {
namespace cuda {

// Finds the Top-K best matches for a batch of queries on the GPU.
// Uses NVIDIA's CUB library for insanely fast segmented radix sorting.
void batch_topk(
    const float* d_distances, // The QxN matrix of all distances
    size_t num_queries,       // Q
    size_t num_vectors,       // N
    size_t k,                 // How many results we want per query
    MetricType metric,        // L2 (smallest) or Inner Product (largest)
    float* d_out_distances,   // Output distances [Q x K]
    size_t* d_out_ids,        // Output IDs [Q x K]
    cudaStream_t stream = 0
);

} 
} 

#endif 
