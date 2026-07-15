#pragma once
#include "vectorforge/compute_backend.h"
#include "vectorforge/distance.h"
#include <queue>
#include <vector>
#include <algorithm>

namespace vectorforge {

class CPUBackend : public ComputeBackend {
public:
    void knn_search(
        const float* queries, size_t num_queries,
        const float* vectors, size_t num_vectors,
        size_t dim, size_t k, MetricType metric,
        float* out_distances, size_t* out_ids) override 
    {
        // For CPU, we just loop and use our AVX2 distance function
        for (size_t q = 0; q < num_queries; ++q) {
            const float* query = queries + (q * dim);
            
            // Max-heap for L2 (we want K smallest), Min-heap for Inner Product (we want K largest)
            auto cmp = [metric](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                if (metric == MetricType::INNER_PRODUCT || metric == MetricType::COSINE) {
                    return a.first > b.first; // Keep largest
                }
                return a.first < b.first; // Keep smallest (L2)
            };
            
            std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(cmp)> pq(cmp);

            for (size_t i = 0; i < num_vectors; ++i) {
                float dist = compute_distance(query, vectors + (i * dim), dim, metric);
                pq.push({dist, i});
                if (pq.size() > k) {
                    pq.pop();
                }
            }

            // Pop into the output arrays (reverse order so best is at index 0)
            size_t output_offset = q * k;
            for (int i = k - 1; i >= 0; --i) {
                out_distances[output_offset + i] = pq.top().first;
                out_ids[output_offset + i] = pq.top().second;
                pq.pop();
            }
        }
    }

    void kmeans_assign(
        const float* vectors, size_t n,
        const float* centroids, size_t num_centroids,
        size_t dim, MetricType metric,
        size_t* assignments) override
    {
        for (size_t i = 0; i < n; ++i) {
            float best_dist = (metric == MetricType::L2) ? 1e30f : -1e30f;
            size_t best_c = 0;
            
            for (size_t c = 0; c < num_centroids; ++c) {
                float dist = compute_distance(vectors + (i * dim), centroids + (c * dim), dim, metric);
                if ((metric == MetricType::L2 && dist < best_dist) || 
                    (metric != MetricType::L2 && dist > best_dist)) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }
    }

    std::string name() const override { return "CPUBackend (AVX2/Scalar)"; }
    bool is_gpu() const override { return false; }
};

} 
