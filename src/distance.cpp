#include "vectorforge/distance.h"
#include <cmath>
#include <immintrin.h>

namespace vectorforge {

float l2_distance_scalar(const float* a, const float* b, size_t dim) {
    float distance = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    // std::sqrt is slow, we can skip it when comparing distances
    return distance; 
}

float dot_product_scalar(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

float cosine_similarity_scalar(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    float mag_a = 0.0f;
    float mag_b = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    
    // Prevent division by zero if a vector is all 0s
    if (mag_a == 0.0f || mag_b == 0.0f) return 0.0f;
    
    return dot / (std::sqrt(mag_a) * std::sqrt(mag_b));
}

#ifdef __AVX2__
// Horizontal reduction of a 256-bit AVX register
inline float reduce_avx2(__m256 v) {
    // Extract upper 128 bits and add to lower 128 bits
    __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    
    // Add adjacent pairs
    v128 = _mm_hadd_ps(v128, v128);
    v128 = _mm_hadd_ps(v128, v128);
    
    // Extract final scalar float
    return _mm_cvtss_f32(v128);
}

float l2_distance_avx2(const float* a, const float* b, size_t dim) {
    float distance = 0.0f;
    size_t i = 0;
    
    // 4 accumulators for loop unrolling (32 floats per iteration)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    
    for (; i + 32 <= dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 diff2 = _mm256_sub_ps(va2, vb2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 diff3 = _mm256_sub_ps(va3, vb3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }
    
    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    // Process remaining elements that are a multiple of 8
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
    }
    
    distance = reduce_avx2(sum0);
    
    // Scalar tail loop
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

float dot_product_avx2(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    size_t i = 0;
    
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    
    for (; i + 32 <= dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }
    
    dot = reduce_avx2(sum0);
    
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    
    // We only unroll by 2 (16 floats per iteration) for Cosine
    // because Cosine uses 3 accumulators + 2 temp registers per unroll chunk.
    // Unrolling by 4 would use 20 registers, causing a "register spill" penalty since AVX2 only has 16 YMM registers.
    __m256 sum_dot0 = _mm256_setzero_ps();
    __m256 sum_dot1 = _mm256_setzero_ps();
    
    __m256 sum_mag_a0 = _mm256_setzero_ps();
    __m256 sum_mag_a1 = _mm256_setzero_ps();
    
    __m256 sum_mag_b0 = _mm256_setzero_ps();
    __m256 sum_mag_b1 = _mm256_setzero_ps();
    
    for (; i + 16 <= dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        sum_dot0 = _mm256_fmadd_ps(va0, vb0, sum_dot0);
        sum_mag_a0 = _mm256_fmadd_ps(va0, va0, sum_mag_a0);
        sum_mag_b0 = _mm256_fmadd_ps(vb0, vb0, sum_mag_b0);
        
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum_dot1 = _mm256_fmadd_ps(va1, vb1, sum_dot1);
        sum_mag_a1 = _mm256_fmadd_ps(va1, va1, sum_mag_a1);
        sum_mag_b1 = _mm256_fmadd_ps(vb1, vb1, sum_mag_b1);
    }
    
    sum_dot0 = _mm256_add_ps(sum_dot0, sum_dot1);
    sum_mag_a0 = _mm256_add_ps(sum_mag_a0, sum_mag_a1);
    sum_mag_b0 = _mm256_add_ps(sum_mag_b0, sum_mag_b1);
    
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum_dot0 = _mm256_fmadd_ps(va, vb, sum_dot0);
        sum_mag_a0 = _mm256_fmadd_ps(va, va, sum_mag_a0);
        sum_mag_b0 = _mm256_fmadd_ps(vb, vb, sum_mag_b0);
    }
    
    float dot = reduce_avx2(sum_dot0);
    float mag_a = reduce_avx2(sum_mag_a0);
    float mag_b = reduce_avx2(sum_mag_b0);
    
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    
    if (mag_a == 0.0f || mag_b == 0.0f) return 0.0f;
    return dot / (std::sqrt(mag_a) * std::sqrt(mag_b));
}
#endif

}
