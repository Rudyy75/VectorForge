#pragma once

#ifdef VECTORFORGE_HAS_CUDA

#include "vectorforge/compute_backend.h"
#include "vectorforge/cuda_utils.h"
#include "vectorforge/distance_kernels.h"
#include "vectorforge/knn_kernels.h"

namespace vectorforge {

// The GPU implementation of our backend.
class CUDABackend : public ComputeBackend {
public:
    CUDABackend() {
        // Create a CUDA stream.
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CUDABackend() override {
        cudaStreamDestroy(stream_);
    }

    void knn_search(
        const float* queries, size_t num_queries,
        const float* vectors, size_t num_vectors,
        size_t dim, size_t k, MetricType metric,
        float* out_distances, size_t* out_ids) override 
    {
        // Allocate GPU memory and upload queries + database vectors
        cuda::CudaDeviceBuffer<float> d_queries(num_queries * dim);
        cuda::CudaDeviceBuffer<float> d_vectors(num_vectors * dim);
        
        d_queries.upload(queries, num_queries * dim, stream_);
        d_vectors.upload(vectors, num_vectors * dim, stream_);

        // Allocate space for the QxN distance matrix
        cuda::CudaDeviceBuffer<float> d_distances(num_queries * num_vectors);

        // Compute all pairwise distances using cuBLAS
        cuda::compute_distances_batch(
            d_queries.data(), d_vectors.data(), d_distances.data(),
            num_queries, num_vectors, dim, metric, stream_
        );

        // Allocate space for the Top-K results
        cuda::CudaDeviceBuffer<float> d_out_distances(num_queries * k);
        cuda::CudaDeviceBuffer<size_t> d_out_ids(num_queries * k);

        // Run our CUB Radix Sort to find the Top-K
        cuda::batch_topk(
            d_distances.data(), num_queries, num_vectors, k, metric,
            d_out_distances.data(), d_out_ids.data(), stream_
        );

        // Download the final tiny Top-K array back to the CPU
        d_out_distances.download(out_distances, num_queries * k, stream_);
        d_out_ids.download(out_ids, num_queries * k, stream_);

        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void kmeans_assign(
        const float* vectors, size_t n,
        const float* centroids, size_t num_centroids,
        size_t dim, MetricType metric,
        size_t* assignments) override
    {
        throw std::runtime_error("CUDABackend::kmeans_assign not implemented yet!");
    }

    std::string name() const override { return "CUDABackend (CUDA/cuBLAS)"; }
    bool is_gpu() const override { return true; }

private:
    cudaStream_t stream_;
};

}

#endif
