#include "vectorforge/distance_kernels.h"
#include "vectorforge/cuda_utils.h"

namespace vectorforge {
namespace cuda {

// Warp level reduction helper
// This function takes a value held by one thread and adds it to the values held by all other threads in the same warp (a group of 32 threads)
__inline__ __device__ float warp_reduce_sum(float val) {
    // __shfl_down_sync lets threads in a warp read each other's variables directly without using slower shared memory
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// The Hand Written L2 Distance Kernel
// This kernel calculates the distance between 1 Query vector and N Database vectors

__global__ void l2_distance_single_query_kernel(
    const float* __restrict__ d_query, 
    const float* __restrict__ d_vectors, 
    float* __restrict__ d_distances, 
    size_t dim) 
{
    // Which database vector is this block responsible for
    size_t vec_idx = blockIdx.x; 
    
    // Allocate shared memory for the query vector (fast cache on the GPU chip)
    extern __shared__ float shared_query[];

    // Threads collaborate to load the query from Global VRAM to Shared VRAM
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        shared_query[i] = d_query[i];
    }
    __syncthreads(); // Wait until all threads finish loading the query

    // Calculate the squared differences
    float local_sum = 0.0f;
    const float* my_db_vector = d_vectors + (vec_idx * dim);

    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = shared_query[i] - my_db_vector[i];
        local_sum += diff * diff;
    }

    // Combine the results from all threads in the block
    // First, combine within each warp (32 threads)
    local_sum = warp_reduce_sum(local_sum);

    // If this thread is the leader of its warp (lane 0), write to shared memory
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    static __shared__ float shared_warp_sums[32]; // Max 1024 threads / 32 = 32 warps
    if (lane == 0) {
        shared_warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Finally, let the very first thread in the block add up the warp sums
    if (threadIdx.x == 0) {
        float final_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; ++i) {
            final_sum += shared_warp_sums[i];
        }
        // Write the final distance to the output array
        d_distances[vec_idx] = final_sum;
    }
}

// C++ Host Wrapper
void compute_distances_single_query(
    const float* d_query, 
    const float* d_vectors, 
    float* d_distances, 
    size_t num_vectors, 
    size_t dim, 
    MetricType metric,
    cudaStream_t stream)
{
    // Launch configuration:
    // N blocks (one for each vector)
    int blocks = num_vectors;
    // 256 threads per block is a good sweet spot for modern GPUs
    int threads = 256; 
    
    // How much shared memory do we need per block (size of the query vector)
    size_t shared_mem_bytes = dim * sizeof(float);

    if (metric == MetricType::L2) {
        // Launch the kernel on the GPU
        // Syntax: kernel<<<blocks, threads, shared_memory, stream>>>(args...)
        l2_distance_single_query_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
            d_query, d_vectors, d_distances, dim
        );
        CUDA_CHECK(cudaGetLastError()); // Check if the launch failed
    } 
    // We will add the Dot Product / Cosine kernels later
}

// Dummy stub for batch cuBLAS (we will implement this fully in the next step)
void compute_distances_batch(
    const float*, const float*, float*, size_t, size_t, size_t, MetricType, cudaStream_t) 
{
    // To be implemented using cuBLAS
}

} 
} 
