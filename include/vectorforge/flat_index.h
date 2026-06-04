#pragma once

#include "vectorforge/types.h"
#include <vector>
#include <shared_mutex>
#include <cstddef>

namespace vectorforge {

class FlatIndex {
public:
    FlatIndex(size_t dim, MetricType metric);
    void add(size_t id, const float* vector);
    std::vector<SearchResult> search(const float* query, size_t k) const;
    size_t size() const;

private:
    size_t dim_;
    MetricType metric_;
    
    // Use single flat array for max CPU cache performance
    std::vector<float> data_;
    std::vector<size_t> ids_;
    
    // Concurrency lock: allows multiple concurrent searches, but exclusive writes
    mutable std::shared_mutex mutex_;
};

}
