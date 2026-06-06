#pragma once

#include "vectorforge/types.h"
#include "vectorforge/thread_pool.h"
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace vectorforge {

class KMeans {
public:
    KMeans(size_t dim, size_t num_clusters, MetricType metric = MetricType::L2);
    
    // Train the clusters using KMeans++. Returns the final centroids.
    // vectors is a flat array of size (num_vectors * dim)
    void train(const float* vectors, size_t num_vectors, ThreadPool& pool, size_t max_iters = 100);
    
    // Get the trained centroids
    const std::vector<float>& get_centroids() const { return centroids_; }

private:
    size_t dim_;
    size_t num_clusters_;
    MetricType metric_;
    std::vector<float> centroids_;
    
    // Initialize using K-Means++ algorithm
    void initialize_kmeans_plus_plus(const float* vectors, size_t num_vectors);
};

}
