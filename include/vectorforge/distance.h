#pragma once

#include <cstddef>

namespace vectorforge {

// (const float) - fastest way to read memory
float l2_distance_scalar(const float* a, const float* b, size_t dim);
float dot_product_scalar(const float* a, const float* b, size_t dim);
float cosine_similarity_scalar(const float* a, const float* b, size_t dim);

// AVX2 SIMD Distance Functions
#ifdef __AVX2__
float l2_distance_avx2(const float* a, const float* b, size_t dim);
float dot_product_avx2(const float* a, const float* b, size_t dim);
float cosine_similarity_avx2(const float* a, const float* b, size_t dim);
#endif

}
