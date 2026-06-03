#include <gtest/gtest.h>
#include "vectorforge/types.h"

TEST(BasicTest, HelloWorld) {
    EXPECT_EQ(1+1, 2);
}

TEST(TypesTest, VectorInitialization) {
    vectorforge::Vector v({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(v.dim, 3);
    EXPECT_FLOAT_EQ(v.data[0], 1.0f);
    EXPECT_FLOAT_EQ(v.data[2], 3.0f);
}

TEST(TypesTest, SearchResultComparison) {
    vectorforge::SearchResult r1{100, 0.5f};
    vectorforge::SearchResult r2{200, 0.8f};
    
    // r1 should be less than r2 since 0.5 < 0.8
    EXPECT_TRUE(r1 < r2);
}