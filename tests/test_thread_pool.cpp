#include <gtest/gtest.h>
#include "vectorforge/thread_pool.h"
#include <atomic>
#include <vector>

using namespace vectorforge;

TEST(ThreadPoolTest, BasicTaskExecution) {
    ThreadPool pool(4);
    
    auto future1 = pool.submit([]() { return 42; });
    auto future2 = pool.submit([](int x, int y) { return x + y; }, 10, 20);
    
    EXPECT_EQ(future1.get(), 42);
    EXPECT_EQ(future2.get(), 30);
}

TEST(ThreadPoolTest, MassTaskExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; ++i) {
        futures.push_back(pool.submit([&counter]() {
            counter++;
        }));
    }
    
    // Wait for all to finish
    for (auto& f : futures) {
        f.wait();
    }
    
    EXPECT_EQ(counter.load(), 1000);
}
