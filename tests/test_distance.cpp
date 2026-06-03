#include <gtest/gtest.h>
#include "vectorforge/distance.h"
#include <vector>

using namespace vectorforge;

TEST(DistanceTest, ScalarL2) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    
    // Differences: -3, -3, -3
    // Squared: 9 + 9 + 9 = 27
    EXPECT_FLOAT_EQ(l2_distance_scalar(a.data(), b.data(), 3), 27.0f);
}

TEST(DistanceTest, ScalarDotProduct) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    
    // Dot: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(dot_product_scalar(a.data(), b.data(), 3), 32.0f);
}

TEST(DistanceTest, ScalarCosine) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    
    // Orthogonal (90 degrees apart) -> Cosine similarity is 0
    EXPECT_FLOAT_EQ(cosine_similarity_scalar(a.data(), b.data(), 2), 0.0f);

    std::vector<float> c = {2.0f, 0.0f};
    
    // Pointing in the exact same direction -> Cosine similarity is 1.0
    EXPECT_FLOAT_EQ(cosine_similarity_scalar(a.data(), c.data(), 2), 1.0f);
}

// AVX2 Verification Tests
#ifdef __AVX2__
void verify_avx2_matches_scalar(size_t dim) {
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    
    for (size_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(dim - i) * 0.1f;
    }
    
    float l2_scalar = l2_distance_scalar(a.data(), b.data(), dim);
    float l2_avx = l2_distance_avx2(a.data(), b.data(), dim);
    // Floating point addition is not associative. AVX2 adds in a different order than Scalar,
    // causing minor precision differences that grow with dimensionality.
    EXPECT_NEAR(l2_scalar, l2_avx, 0.05f) << "L2 failed at dim " << dim;
    
    float dot_scalar = dot_product_scalar(a.data(), b.data(), dim);
    float dot_avx = dot_product_avx2(a.data(), b.data(), dim);
    EXPECT_NEAR(dot_scalar, dot_avx, 0.05f) << "Dot Product failed at dim " << dim;
    
    float cos_scalar = cosine_similarity_scalar(a.data(), b.data(), dim);
    float cos_avx = cosine_similarity_avx2(a.data(), b.data(), dim);
    // Cosine similarity is always between -1 and 1, so 1e-4f is still a safe threshold
    EXPECT_NEAR(cos_scalar, cos_avx, 1e-4f) << "Cosine failed at dim " << dim;
}

TEST(DistanceTest, AVX2MatchesScalar) {
    std::vector<size_t> dims = {1, 7, 8, 15, 16, 128, 300};
    for (size_t dim : dims) {
        verify_avx2_matches_scalar(dim);
    }
}
#endif
