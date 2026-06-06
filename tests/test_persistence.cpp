#include <gtest/gtest.h>
#include "vectorforge/flat_index.h"
#include "vectorforge/ivf_index.h"
#include <random>
#include <cstdio> 

using namespace vectorforge;

class PersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 64;
        num_vectors = 500;
        
        data.resize(num_vectors * dim);
        ids.resize(num_vectors);
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < num_vectors; ++i) {
            ids[i] = i;
            for (size_t d = 0; d < dim; ++d) {
                data[i * dim + d] = dist(rng);
            }
        }
    }
    
    void TearDown() override {
        // Clean up temp files
        std::remove("test_flat.index");
        std::remove("test_ivf.index");
    }

    size_t dim;
    size_t num_vectors;
    std::vector<float> data;
    std::vector<size_t> ids;
};

TEST_F(PersistenceTest, FlatIndexSaveLoad) {
    FlatIndex original_index(dim, MetricType::L2);
    for (size_t i = 0; i < num_vectors; ++i) {
        original_index.add(ids[i], data.data() + i * dim);
    }
    
    EXPECT_NO_THROW(original_index.save("test_flat.index"));
    
    FlatIndex loaded_index(1, MetricType::Cosine); // Dummy initialization
    EXPECT_NO_THROW(loaded_index.load("test_flat.index"));
    
    EXPECT_EQ(loaded_index.size(), num_vectors);
    
    // Verify search results are identical
    auto orig_res = original_index.search(data.data(), 5);
    auto load_res = loaded_index.search(data.data(), 5);
    
    ASSERT_EQ(orig_res.size(), load_res.size());
    for (size_t i = 0; i < orig_res.size(); ++i) {
        EXPECT_EQ(orig_res[i].id, load_res[i].id);
        EXPECT_FLOAT_EQ(orig_res[i].distance, load_res[i].distance);
    }
}

TEST_F(PersistenceTest, IVFIndexSaveLoad) {
    ThreadPool pool(4);
    IVFIndex original_index(dim, 5, MetricType::L2);
    
    original_index.train(data.data(), num_vectors, pool);
    original_index.add(data.data(), ids.data(), num_vectors, &pool);
    
    EXPECT_NO_THROW(original_index.save("test_ivf.index"));
    
    IVFIndex loaded_index(1, 1, MetricType::Cosine); // Dummy initialization
    EXPECT_NO_THROW(loaded_index.load("test_ivf.index"));
    
    EXPECT_TRUE(loaded_index.is_trained());
    EXPECT_EQ(loaded_index.size(), num_vectors);
    
    // Set nprobe
    original_index.set_nprobe(2);
    loaded_index.set_nprobe(2);
    
    // Verify search results are identical
    auto orig_res = original_index.search(data.data(), 1, 5, &pool);
    auto load_res = loaded_index.search(data.data(), 1, 5, &pool);
    
    ASSERT_EQ(orig_res.second.size(), load_res.second.size());
    for (size_t i = 0; i < orig_res.second.size(); ++i) {
        EXPECT_EQ(orig_res.second[i], load_res.second[i]);
        EXPECT_FLOAT_EQ(orig_res.first[i], load_res.first[i]);
    }
}
