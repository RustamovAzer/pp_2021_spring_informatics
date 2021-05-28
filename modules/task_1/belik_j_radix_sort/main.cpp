// Copyright 2021 Belik Julia
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "./RadixSortB.h"

TEST(Radix_Sort_Merge_Batcher, Test_Vector_OneEl) {
    std::vector<double> vec;
    const int n = 1;
    vec = Vector(n, -10.0, 10.0);
    std::vector<double> v1 = RadixSort(vec);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Vector_Positive) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, 0.0, 10000.0);
    std::vector<double> v1 = RadixSort(vec);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Vector_Negative) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, -10000.0, 0.0);
    std::vector<double> v1 = RadixSort(vec);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Vector_10El) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, -10000.0, 10000.0);
    std::vector<double> v1 = RadixSort(vec);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Vector_100El) {
    std::vector<double> vec;
    const int n = 100;
    vec = Vector(n, -10000.0, 10000.0);
    std::vector<double> v1 = RadixSort(vec);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
