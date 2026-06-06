#pragma once

#include "vectorforge/types.h"
#include "vectorforge/thread_pool.h"
#include "vectorforge/kmeans.h"
#include <vector>
#include <cstddef>
#include <shared_mutex>
#include <memory>
#include <queue>
#include <utility>

namespace vectorforge {

class IVFIndex {
public:
    IVFIndex(size_t dim, size_t nlist, MetricType metric = MetricType::L2);
    
    // Train the IVFIndex to find the nlist cluster centroids before any addition of data
    void train(const float* vectors, size_t n, ThreadPool& pool);
    bool is_trained() const { return is_trained_; }
    
    // Set how many lists to search during a query (default is 1)
    void set_nprobe(size_t nprobe) { nprobe_ = nprobe; }
    
    // Add vectors to the index. If pool is provided, parallelizes the insertion.
    void add(const float* vectors, const size_t* ids, size_t n, ThreadPool* pool = nullptr);
    
    // Search the index. 
    // Returns a pair of vectors containing the distances and the IDs.
    // distances will be size (num_queries * k)
    // labels will be size (num_queries * k)
    std::pair<std::vector<float>, std::vector<size_t>> search(
        const float* queries, size_t num_queries, size_t k, ThreadPool* pool = nullptr) const;
        
    size_t size() const;

private:
    size_t dim_;
    size_t nlist_;
    size_t nprobe_;
    MetricType metric_;
    bool is_trained_;
    
    std::vector<float> centroids_;
    
    struct Bucket {
        std::vector<float> data;
        std::vector<size_t> ids;
        mutable std::shared_mutex mtx; // Allows concurrent read or exclusive write
    };
    
    // Array of nlist buckets
    std::vector<std::unique_ptr<Bucket>> buckets_;
    
    size_t find_closest_centroid(const float* vec) const;
    std::vector<size_t> find_nprobe_closest_centroids(const float* vec) const;
};

}
