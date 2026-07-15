#include "vectorforge/knn_kernels.h"
#include "vectorforge/cuda_utils.h"
#include <cub/cub.cuh>
#include <iostream>

namespace vectorforge {
namespace cuda {

// Helper kernel: Fills an array with 0, 1, 2... N-1 repeatedly for each query
// so that when we sort the distances, we know which vector ID it belonged to
__global__ void init_ids_kernel(size_t* d_ids, size_t num_queries, size_t num_vectors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_queries * num_vectors;
    if (idx < total) {
        d_ids[idx] = idx % num_vectors;
    }
}

// Helper kernel: Sets up the segment offsets for CUB
// This tells the sorter where one query ends and the next begins
__global__ void init_offsets_kernel(int* d_offsets, size_t num_queries, size_t num_vectors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_queries) { 
        d_offsets[idx] = idx * num_vectors;
    }
}

// Helper kernel: Copies only the top K results from the huge sorted arrays into our tiny output arrays
__global__ void extract_topk_kernel(
    const float* d_sorted_distances,
    const size_t* d_sorted_ids,
    float* d_out_distances,
    size_t* d_out_ids,
    size_t num_queries,
    size_t num_vectors,
    size_t k) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_k = num_queries * k;
    if (idx < total_k) {
        size_t q = idx / k;         // Which query is this?
        size_t rank = idx % k;      // Which rank (0 to K-1) is this?
        
        // Grab the value from the giant sorted matrix
        size_t sorted_idx = q * num_vectors + rank;
        d_out_distances[idx] = d_sorted_distances[sorted_idx];
        d_out_ids[idx] = d_sorted_ids[sorted_idx];
    }
}

void batch_topk(
    const float* d_distances,
    size_t num_queries,
    size_t num_vectors,
    size_t k,
    MetricType metric,
    float* d_out_distances,
    size_t* d_out_ids,
    cudaStream_t stream)
{
    size_t total_elements = num_queries * num_vectors;

    // Allocate temporary memory for the original IDs
    CudaDeviceBuffer<size_t> d_ids(total_elements);
    
    // Allocate temporary memory for the SORTED distances and IDs
    CudaDeviceBuffer<float> d_sorted_distances(total_elements);
    CudaDeviceBuffer<size_t> d_sorted_ids(total_elements);

    // Initialize the IDs array (0 to N-1 for each query)
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    init_ids_kernel<<<blocks, threads, 0, stream>>>(d_ids.data(), num_queries, num_vectors);
    CUDA_CHECK(cudaGetLastError());

    // Setup segment offsets for CUB Segmented Sort
    CudaDeviceBuffer<int> d_offsets(num_queries + 1);
    int offset_blocks = (num_queries + 1 + threads - 1) / threads;
    init_offsets_kernel<<<offset_blocks, threads, 0, stream>>>(d_offsets.data(), num_queries, num_vectors);
    CUDA_CHECK(cudaGetLastError());

    // Ask CUB how much temporary workspace memory it needs to do the sort
    size_t temp_storage_bytes = 0;
    if (metric == MetricType::L2) {
        // L2 distance: Smallest is best, so sort Ascending
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            d_distances, d_sorted_distances.data(),
            d_ids.data(), d_sorted_ids.data(),
            total_elements, num_queries,
            d_offsets.data(), d_offsets.data() + 1,
            0, sizeof(float)*8, stream
        );
    } else {
        // Inner Product / Cosine: Largest is best, so sort Descending
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            d_distances, d_sorted_distances.data(),
            d_ids.data(), d_sorted_ids.data(),
            total_elements, num_queries,
            d_offsets.data(), d_offsets.data() + 1,
            0, sizeof(float)*8, stream
        );
    }

    // Allocate the workspace memory CUB asked for
    CudaDeviceBuffer<uint8_t> d_temp_storage(temp_storage_bytes);

    // Run the massively parallel Radix Sort
    if (metric == MetricType::L2) {
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage.data(), temp_storage_bytes,
            d_distances, d_sorted_distances.data(),
            d_ids.data(), d_sorted_ids.data(),
            total_elements, num_queries,
            d_offsets.data(), d_offsets.data() + 1,
            0, sizeof(float)*8, stream
        );
    } else {
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_temp_storage.data(), temp_storage_bytes,
            d_distances, d_sorted_distances.data(),
            d_ids.data(), d_sorted_ids.data(),
            total_elements, num_queries,
            d_offsets.data(), d_offsets.data() + 1,
            0, sizeof(float)*8, stream
        );
    }
    CUDA_CHECK(cudaGetLastError());

    // Extract only the Top K results for each query and put them in the output arrays
    size_t total_k = num_queries * k;
    int k_blocks = (total_k + threads - 1) / threads;
    extract_topk_kernel<<<k_blocks, threads, 0, stream>>>(
        d_sorted_distances.data(),
        d_sorted_ids.data(),
        d_out_distances,
        d_out_ids,
        num_queries,
        num_vectors,
        k
    );
    CUDA_CHECK(cudaGetLastError());
}

} 
} 
