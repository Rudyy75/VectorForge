#pragma once
#include "vectorforge/types.h"
#include <cstddef>
#include <string>
#include <memory>

namespace vectorforge {

// Any backend (CPU, CUDA, or future ones like Apple Metal) must follow these rules.
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Brute-force kNN: query batch -> results
    virtual void knn_search(
        const float* queries, size_t num_queries,
        const float* vectors, size_t num_vectors,
        size_t dim, size_t k, MetricType metric,
        float* out_distances, size_t* out_ids) = 0;

    // K-Means assignment step: assign N vectors to K centroids
    virtual void kmeans_assign(
        const float* vectors, size_t n,
        const float* centroids, size_t num_centroids,
        size_t dim, MetricType metric,
        size_t* assignments) = 0;

    virtual std::string name() const = 0;
    virtual bool is_gpu() const = 0;
};

// Factory function to get the right backend
std::unique_ptr<ComputeBackend> create_backend(const std::string& device = "auto");

}
