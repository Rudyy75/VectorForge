#include <gtest/gtest.h>
#include "vectorforge/kmeans.h"
#include <vector>
#include <cmath>

using namespace vectorforge;

TEST(KMeansTest, ThreeGaussianClusters) {
    // We will generate 3 distinct blobs of 2D data:
    // Blob 1 exactly at (0,0)
    // Blob 2 exactly at (10,10)
    // Blob 3 exactly at (-10, -10)
    
    std::vector<float> data;
    
    auto add_blob = [&](float x, float y) {
        for (int i = 0; i < 50; ++i) {
            data.push_back(x);
            data.push_back(y);
        }
    };
    
    add_blob(0.0f, 0.0f);
    add_blob(10.0f, 10.0f);
    add_blob(-10.0f, -10.0f);
    
    ThreadPool pool(4);
    KMeans kmeans(2, 3, MetricType::L2);
    
    // 150 vectors total
    kmeans.train(data.data(), 150, pool);
    
    const auto& centroids = kmeans.get_centroids();
    ASSERT_EQ(centroids.size(), 6); // 3 clusters * 2 dims
    
    // Check if the centroids match our 3 blobs (in any order)
    bool found_blob1 = false;
    bool found_blob2 = false;
    bool found_blob3 = false;
    
    for (size_t i = 0; i < 3; ++i) {
        float cx = centroids[i*2];
        float cy = centroids[i*2 + 1];
        
        if (std::abs(cx - 0.0f) < 0.1f && std::abs(cy - 0.0f) < 0.1f) found_blob1 = true;
        if (std::abs(cx - 10.0f) < 0.1f && std::abs(cy - 10.0f) < 0.1f) found_blob2 = true;
        if (std::abs(cx + 10.0f) < 0.1f && std::abs(cy + 10.0f) < 0.1f) found_blob3 = true;
    }
    
    EXPECT_TRUE(found_blob1);
    EXPECT_TRUE(found_blob2);
    EXPECT_TRUE(found_blob3);
}
