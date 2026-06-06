#include "vectorforge/ivf_index.h"
#include "vectorforge/distance.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace vectorforge {

IVFIndex::IVFIndex(size_t dim, size_t nlist, MetricType metric)
    : dim_(dim), nlist_(nlist), nprobe_(1), metric_(metric), is_trained_(false) {
    if (dim == 0 || nlist == 0) {
        throw std::invalid_argument("Dimension and nlist must be > 0");
    }
    
    for (size_t i = 0; i < nlist_; ++i) {
        buckets_.push_back(std::make_unique<Bucket>());
    }
}

void IVFIndex::train(const float* vectors, size_t n, ThreadPool& pool) {
    if (n < nlist_) {
        throw std::invalid_argument("Number of training vectors must be >= nlist");
    }
    
    KMeans kmeans(dim_, nlist_, metric_);
    kmeans.train(vectors, n, pool, 100);
    
    centroids_ = kmeans.get_centroids();
    is_trained_ = true;
}

size_t IVFIndex::find_closest_centroid(const float* vec) const {
    float best_dist = std::numeric_limits<float>::max();
    size_t best_idx = 0;
    
    for (size_t i = 0; i < nlist_; ++i) {
        float dist = 0.0f;
        const float* centroid = centroids_.data() + (i * dim_);
        
#ifdef __AVX2__
        if (metric_ == MetricType::L2) {
            dist = l2_distance_avx2(vec, centroid, dim_);
        } else {
            dist = dot_product_avx2(vec, centroid, dim_);
        }
#else
        if (metric_ == MetricType::L2) {
            dist = l2_distance_scalar(vec, centroid, dim_);
        } else {
            dist = dot_product_scalar(vec, centroid, dim_);
        }
#endif
        
        if (metric_ != MetricType::L2) {
            dist = -dist; // Convert similarity to distance
        }
        
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

std::vector<size_t> IVFIndex::find_nprobe_closest_centroids(const float* vec) const {
    // Priority queue to maintain Top-K (nprobe) closest centroids
    // We want the smallest distances, so we use a Max-Heap (std::less) to keep the largest of the small distances at the top
    using Element = std::pair<float, size_t>;
    std::priority_queue<Element, std::vector<Element>, std::less<Element>> pq;
    
    for (size_t i = 0; i < nlist_; ++i) {
        float dist = 0.0f;
        const float* centroid = centroids_.data() + (i * dim_);
        
#ifdef __AVX2__
        if (metric_ == MetricType::L2) {
            dist = l2_distance_avx2(vec, centroid, dim_);
        } else {
            dist = dot_product_avx2(vec, centroid, dim_);
        }
#else
        if (metric_ == MetricType::L2) {
            dist = l2_distance_scalar(vec, centroid, dim_);
        } else {
            dist = dot_product_scalar(vec, centroid, dim_);
        }
#endif
        
        if (metric_ != MetricType::L2) {
            dist = -dist; 
        }
        
        if (pq.size() < nprobe_) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, i});
        }
    }
    
    std::vector<size_t> result;
    result.reserve(pq.size());
    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }
    std::reverse(result.begin(), result.end());
    
    return result;
}

void IVFIndex::add(const float* vectors, const size_t* ids, size_t n, ThreadPool* pool) {
    if (!is_trained_) {
        throw std::runtime_error("Cannot add vectors to an untrained IVFIndex");
    }
    
    auto add_chunk = [this, vectors, ids](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const float* vec = vectors + (i * dim_);
            size_t bucket_idx = find_closest_centroid(vec);
            
            // Re-normalize if Cosine metric
            std::vector<float> normalized_vec(dim_);
            if (metric_ == MetricType::Cosine) {
                float mag = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    mag += vec[d] * vec[d];
                }
                mag = std::sqrt(mag);
                if (mag == 0.0f) mag = 1.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    normalized_vec[d] = vec[d] / mag;
                }
                vec = normalized_vec.data();
            }
            
            // Lock this specific bucket and add
            Bucket* b = buckets_[bucket_idx].get();
            std::unique_lock<std::shared_mutex> lock(b->mtx);
            b->data.insert(b->data.end(), vec, vec + dim_);
            b->ids.push_back(ids[i]);
        }
    };

    if (pool) {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        std::vector<std::future<void>> futures;
        size_t chunk_size = (n + num_threads - 1) / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            if (start >= end) break;
            
            futures.push_back(pool->submit(add_chunk, start, end));
        }
        
        for (auto& f : futures) {
            f.wait();
        }
    } else {
        add_chunk(0, n);
    }
}

size_t IVFIndex::size() const {
    size_t total = 0;
    for (size_t i = 0; i < nlist_; ++i) {
        std::shared_lock<std::shared_mutex> lock(buckets_[i]->mtx);
        total += buckets_[i]->ids.size();
    }
    return total;
}

std::pair<std::vector<float>, std::vector<size_t>> IVFIndex::search(
    const float* queries, size_t num_queries, size_t k, ThreadPool* pool) const {
    
    if (!is_trained_) {
        throw std::runtime_error("Cannot search untrained IVFIndex");
    }
    
    std::vector<float> all_distances(num_queries * k, 
        (metric_ == MetricType::L2) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max());
    std::vector<size_t> all_ids(num_queries * k, static_cast<size_t>(-1));

    auto search_query = [this, k](const float* query, float* out_dist, size_t* out_ids) {
        // Preprocess Cosine
        std::vector<float> norm_query(dim_);
        const float* q = query;
        if (metric_ == MetricType::Cosine) {
            float mag = 0.0f;
            for (size_t d = 0; d < dim_; ++d) mag += query[d] * query[d];
            mag = std::sqrt(mag);
            if (mag == 0.0f) mag = 1.0f;
            for (size_t d = 0; d < dim_; ++d) norm_query[d] = query[d] / mag;
            q = norm_query.data();
        }
        
        std::vector<size_t> target_buckets = find_nprobe_closest_centroids(q);
        
        using Element = std::pair<float, size_t>;
        // L2: we want K smallest distances - Max-Heap keeps largest at top
        // Dot/Cosine: we want K largest similarities - Min-Heap keeps smallest at top
        auto cmp = [this](const Element& left, const Element& right) {
            if (metric_ == MetricType::L2) return left.first < right.first; // Max-Heap
            return left.first > right.first; // Min-Heap
        };
        
        std::priority_queue<Element, std::vector<Element>, decltype(cmp)> pq(cmp);
        
        for (size_t bucket_idx : target_buckets) {
            Bucket* b = buckets_[bucket_idx].get();
            std::shared_lock<std::shared_mutex> lock(b->mtx);
            
            size_t num_in_bucket = b->ids.size();
            for (size_t i = 0; i < num_in_bucket; ++i) {
                const float* vec = b->data.data() + (i * dim_);
                float dist = 0.0f;
#ifdef __AVX2__
                if (metric_ == MetricType::L2) dist = l2_distance_avx2(q, vec, dim_);
                else dist = dot_product_avx2(q, vec, dim_);
#else
                if (metric_ == MetricType::L2) dist = l2_distance_scalar(q, vec, dim_);
                else dist = dot_product_scalar(q, vec, dim_);
#endif
                
                if (pq.size() < k) {
                    pq.push({dist, b->ids[i]});
                } else {
                    bool should_push = (metric_ == MetricType::L2) ? (dist < pq.top().first) : (dist > pq.top().first);
                    if (should_push) {
                        pq.pop();
                        pq.push({dist, b->ids[i]});
                    }
                }
            }
        }
        
        // Extract from PQ
        std::vector<Element> temp;
        temp.reserve(pq.size());
        while (!pq.empty()) {
            temp.push_back(pq.top());
            pq.pop();
        }
        
        std::reverse(temp.begin(), temp.end());
        
        size_t count = temp.size();
        for (size_t i = 0; i < k; ++i) {
            if (i < count) {
                out_dist[i] = temp[i].first;
                out_ids[i] = temp[i].second;
            } else {
                // Pad if not enough results
                out_dist[i] = (metric_ == MetricType::L2) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
                out_ids[i] = static_cast<size_t>(-1);
            }
        }
    };
    
    if (pool) {
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < num_queries; ++i) {
            float* out_dist = all_distances.data() + (i * k);
            size_t* out_ids = all_ids.data() + (i * k);
            futures.push_back(pool->submit(search_query, queries + (i * dim_), out_dist, out_ids));
        }
        for (auto& f : futures) f.wait();
    } else {
        for (size_t i = 0; i < num_queries; ++i) {
            float* out_dist = all_distances.data() + (i * k);
            size_t* out_ids = all_ids.data() + (i * k);
            search_query(queries + (i * dim_), out_dist, out_ids);
        }
    }
    
    return {all_distances, all_ids};
}

}
