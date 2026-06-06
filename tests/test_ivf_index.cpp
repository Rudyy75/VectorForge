#include <gtest/gtest.h>
#include "vectorforge/ivf_index.h"
#include <random>

using namespace vectorforge;

TEST(IVFIndexTest, BasicTrainAddSearch) {
    size_t dim = 128;
    size_t nlist = 5;
    size_t num_vectors = 1000;
    
    std::vector<float> data(num_vectors * dim);
    std::vector<size_t> ids(num_vectors);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
        for (size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = dist(rng);
        }
    }
    
    ThreadPool pool(4);
    IVFIndex index(dim, nlist, MetricType::L2);
    
    // Train
    EXPECT_NO_THROW(index.train(data.data(), num_vectors, pool));
    EXPECT_TRUE(index.is_trained());
    
    // Add
    EXPECT_NO_THROW(index.add(data.data(), ids.data(), num_vectors, &pool));
    EXPECT_EQ(index.size(), num_vectors);
    
    // Search
    index.set_nprobe(2); // Search top 2 out of 5 buckets
    
    // Query with the first vector in our dataset, it should find itself as the top result
    auto result = index.search(data.data(), 1, 5, &pool);
    const auto& distances = result.first;
    const auto& labels = result.second;
    
    ASSERT_EQ(labels.size(), 5);
    EXPECT_EQ(labels[0], 0); // The closest vector should be exactly itself
    EXPECT_NEAR(distances[0], 0.0f, 1e-5); // Distance to itself is 0
}
