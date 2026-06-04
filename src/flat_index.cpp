#include "vectorforge/flat_index.h"
#include "vectorforge/distance.h"
#include <cmath>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <mutex>

namespace vectorforge {

FlatIndex::FlatIndex(size_t dim, MetricType metric) 
    : dim_(dim), metric_(metric) {
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
}

size_t FlatIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return ids_.size();
}

void FlatIndex::add(size_t id, const float* vector) {
    // Unique lock for exclusive write access
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    ids_.push_back(id);
    
    if (metric_ == MetricType::Cosine) {
        // Pre-normalize the vector to length 1.0 at insertion time
        float mag = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            mag += vector[i] * vector[i];
        }
        mag = std::sqrt(mag);
        
        for (size_t i = 0; i < dim_; ++i) {
            if (mag > 0.0f) {
                data_.push_back(vector[i] / mag);
            } else {
                data_.push_back(0.0f);
            }
        }
    } else {
        // For L2 and Dot Product, store exactly as-is
        data_.insert(data_.end(), vector, vector + dim_);
    }
}

std::vector<SearchResult> FlatIndex::search(const float* query, size_t k) const {
    if (k == 0) return {};
    
    // Shared lock allowing multiple concurrent readers
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t num_vectors = ids_.size();
    if (num_vectors == 0) return {};
    
    // If Cosine, we must also normalize the query before searching
    std::vector<float> normalized_query;
    const float* search_query = query;
    if (metric_ == MetricType::Cosine) {
        normalized_query.resize(dim_);
        float mag = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            mag += query[i] * query[i];
        }
        mag = std::sqrt(mag);
        for (size_t i = 0; i < dim_; ++i) {
            normalized_query[i] = mag > 0.0f ? query[i] / mag : 0.0f;
        }
        search_query = normalized_query.data();
    }

    // L2: Smaller is better (Max-Heap keeps the K smallest)
    // Dot/Cosine: Larger is better (Min-Heap keeps the K largest)
    bool is_l2 = (metric_ == MetricType::L2);
    
    auto cmp = [is_l2](const SearchResult& a, const SearchResult& b) {
        if (is_l2) {
            return a.distance < b.distance; // Max-Heap
        } else {
            return a.distance > b.distance; // Min-Heap
        }
    };
    
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> pq(cmp);
    
    // Scan all vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        const float* db_vec = data_.data() + (i * dim_);
        float dist = 0.0f;
        
#ifdef __AVX2__
        if (is_l2) {
            dist = l2_distance_avx2(search_query, db_vec, dim_);
        } else {
            // Cosine pre-normalization means we just run Dot-Product
            dist = dot_product_avx2(search_query, db_vec, dim_);
        }
#else
        if (is_l2) {
            dist = l2_distance_scalar(search_query, db_vec, dim_);
        } else {
            dist = dot_product_scalar(search_query, db_vec, dim_);
        }
#endif

        if (pq.size() < k) {
            pq.push({ids_[i], dist});
        } else {
            bool should_replace = is_l2 ? (dist < pq.top().distance) : (dist > pq.top().distance);
            if (should_replace) {
                pq.pop();
                pq.push({ids_[i], dist});
            }
        }
    }
    
    // Extract results
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
        results.push_back(pq.top());
        pq.pop();
    }
    
    // Heap pops worst-of-the-best first, so reverse to get best-first
    std::reverse(results.begin(), results.end());
    
    return results;
}

}
