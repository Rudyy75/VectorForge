#include "vectorforge/kmeans.h"
#include "vectorforge/distance.h"
#include <random>
#include <limits>
#include <iostream>

namespace vectorforge {

KMeans::KMeans(size_t dim, size_t num_clusters, MetricType metric)
    : dim_(dim), num_clusters_(num_clusters), metric_(metric) {
    if (dim == 0 || num_clusters == 0) {
        throw std::invalid_argument("Dimension and num_clusters must be > 0");
    }
}

void KMeans::initialize_kmeans_plus_plus(const float* vectors, size_t num_vectors) {
    centroids_.clear();
    centroids_.reserve(num_clusters_ * dim_);
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> dist(0, num_vectors - 1);
    
    // Pick first centroid randomly
    size_t first_idx = dist(rng);
    centroids_.insert(centroids_.end(), 
                      vectors + (first_idx * dim_), 
                      vectors + ((first_idx + 1) * dim_));
                      
    std::vector<float> min_distances(num_vectors, std::numeric_limits<float>::max());
    
    // Pick remaining centroids based on D(x)^2 probability
    for (size_t k = 1; k < num_clusters_; ++k) {
        float sum_dist = 0.0f;
        const float* latest_centroid = centroids_.data() + ((k - 1) * dim_);
        
        for (size_t i = 0; i < num_vectors; ++i) {
            const float* vec = vectors + (i * dim_);
            
            float d = 0.0f;
#ifdef __AVX2__
            d = (metric_ == MetricType::L2) ? 
                l2_distance_avx2(vec, latest_centroid, dim_) : 
                dot_product_avx2(vec, latest_centroid, dim_);
#else
            d = (metric_ == MetricType::L2) ? 
                l2_distance_scalar(vec, latest_centroid, dim_) : 
                dot_product_scalar(vec, latest_centroid, dim_);
#endif
            
            if (metric_ != MetricType::L2) {
                // If metric is Cosine (pre-normalized), max dot product is 1.0
                // Convert similarity to distance: distance = 1.0 - dot_product
                d = 1.0f - d; 
                if (d < 0.0f) d = 0.0f;
            }
            
            // D(x)^2
            d = d * d; 
            
            if (d < min_distances[i]) {
                min_distances[i] = d;
            }
            sum_dist += min_distances[i];
        }
        
        std::uniform_real_distribution<float> float_dist(0.0f, sum_dist);
        float r = float_dist(rng);
        
        size_t next_idx = num_vectors - 1;
        float current_sum = 0.0f;
        for (size_t i = 0; i < num_vectors; ++i) {
            current_sum += min_distances[i];
            if (current_sum >= r) {
                next_idx = i;
                break;
            }
        }
        
        centroids_.insert(centroids_.end(), 
                          vectors + (next_idx * dim_), 
                          vectors + ((next_idx + 1) * dim_));
    }
}

void KMeans::train(const float* vectors, size_t num_vectors, ThreadPool& pool, size_t max_iters) {
    if (num_vectors < num_clusters_) {
        throw std::invalid_argument("Number of vectors must be >= number of clusters");
    }
    
    // Smart Initialization
    initialize_kmeans_plus_plus(vectors, num_vectors);
    
    std::vector<size_t> assignments(num_vectors, std::numeric_limits<size_t>::max());
    bool changed = true;
    size_t iter = 0;
    
    // Distribute work across 4 threads (or hardware concurrency)
    size_t num_threads = 4;
    
    while (changed && iter < max_iters) {
        changed = false;
        
        // LLOYD ITERATION STEP 1: Assignment (Parallel)
        std::vector<std::future<size_t>> futures;
        size_t chunk_size = (num_vectors + num_threads - 1) / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, num_vectors);
            
            if (start >= end) break;
            
            futures.push_back(pool.submit([this, vectors, start, end, &assignments]() {
                size_t local_changes = 0;
                for (size_t i = start; i < end; ++i) {
                    const float* vec = vectors + (i * dim_);
                    
                    float best_dist = std::numeric_limits<float>::max();
                    size_t best_cluster = 0;
                    
                    for (size_t k = 0; k < num_clusters_; ++k) {
                        const float* centroid = centroids_.data() + (k * dim_);
                        float dist = 0.0f;
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
                            dist = -dist; // Convert similarity to distance (larger is better, so negate)
                        }
                        
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_cluster = k;
                        }
                    }
                    
                    if (assignments[i] != best_cluster) {
                        assignments[i] = best_cluster;
                        local_changes++;
                    }
                }
                return local_changes;
            }));
        }
        
        size_t total_changes = 0;
        for (auto& f : futures) {
            total_changes += f.get();
        }
        
        if (total_changes == 0) {
            break; // We have fully converged
        }
        
        // LLOYD ITERATION STEP 2: Update Centroids
        std::vector<float> new_centroids(num_clusters_ * dim_, 0.0f);
        std::vector<size_t> cluster_counts(num_clusters_, 0);
        
        for (size_t i = 0; i < num_vectors; ++i) {
            size_t c = assignments[i];
            cluster_counts[c]++;
            const float* vec = vectors + (i * dim_);
            for (size_t d = 0; d < dim_; ++d) {
                new_centroids[c * dim_ + d] += vec[d];
            }
        }
        
        // Average the sums
        for (size_t k = 0; k < num_clusters_; ++k) {
            if (cluster_counts[k] > 0) {
                for (size_t d = 0; d < dim_; ++d) {
                    centroids_[k * dim_ + d] = new_centroids[k * dim_ + d] / cluster_counts[k];
                }
            }
            
            // If Cosine, the centroid must be re-normalized to length 1.0
            if (metric_ == MetricType::Cosine && cluster_counts[k] > 0) {
                float mag = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    float val = centroids_[k * dim_ + d];
                    mag += val * val;
                }
                mag = std::sqrt(mag);
                if (mag > 0.0f) {
                    for (size_t d = 0; d < dim_; ++d) {
                        centroids_[k * dim_ + d] /= mag;
                    }
                }
            }
        }
        
        changed = true;
        iter++;
    }
}

} 
