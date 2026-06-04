#include <gtest/gtest.h>
#include "vectorforge/flat_index.h"
#include <vector>
#include <thread>
#include <random>

using namespace vectorforge;

TEST(FlatIndexTest, L2SearchCorrectness) {
    FlatIndex index(2, MetricType::L2);
    
    std::vector<float> v0 = {0.0f, 0.0f};
    std::vector<float> v1 = {1.0f, 1.0f};
    std::vector<float> v2 = {2.0f, 2.0f};
    
    index.add(0, v0.data());
    index.add(1, v1.data());
    index.add(2, v2.data());
    
    std::vector<float> query = {0.1f, 0.1f};
    auto results = index.search(query.data(), 2);
    
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 0); // 0.0 is closest to 0.1
    EXPECT_EQ(results[1].id, 1); // 1.0 is next
}

TEST(FlatIndexTest, CosinePreNormalization) {
    FlatIndex index(2, MetricType::Cosine);
    
    std::vector<float> v0 = {1.0f, 0.0f}; // points right
    std::vector<float> v1 = {0.0f, 1.0f}; // points up
    std::vector<float> v2 = {2.0f, 0.0f}; // points right, but longer
    
    index.add(0, v0.data());
    index.add(1, v1.data());
    index.add(2, v2.data());
    
    std::vector<float> query = {5.0f, 0.0f}; // points right
    auto results = index.search(query.data(), 2);
    
    ASSERT_EQ(results.size(), 2);
    // Both 0 and 2 point in the exact same direction, so they should be tied at 1.0 similarity
    EXPECT_TRUE(results[0].id == 0 || results[0].id == 2);
    EXPECT_FLOAT_EQ(results[0].distance, 1.0f);
}

TEST(FlatIndexTest, ThreadSafety) {
    FlatIndex index(128, MetricType::L2);
    
    auto writer = [&]() {
        std::vector<float> v(128, 1.0f);
        for (size_t i = 0; i < 1000; ++i) {
            index.add(i, v.data());
        }
    };
    
    auto reader = [&]() {
        std::vector<float> q(128, 1.0f);
        for (size_t i = 0; i < 500; ++i) {
            auto res = index.search(q.data(), 5);
        }
    };
    
    std::vector<std::thread> threads;
    threads.emplace_back(writer);
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(reader);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(index.size(), 1000);
}
