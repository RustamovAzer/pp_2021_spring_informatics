// Copyright 2021 Belik Julia
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "./RadixSortB.h"

TEST(Radix_Sort_Merge_Batcher, Test_Sort) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, -10000.0, 10000.0);
    double* tmp1 = new double[n];
    double* tmp2 = new double[n];
    for (int i = 0; i < n; i++)
        tmp1[i] = vec[i];
    RadixSort(tmp1, n, tmp2);
    std::vector<double> v1(n);
    for (int i = 0; i < n; i++)
        v1[i] = tmp2[i];
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Sort_Par_1thr) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, -10.0, 10.0);
    std::vector<double> v1 = MergeBatcherPar(vec, n, 1);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Sort_Seq_1thr) {
    std::vector<double> vec;
    const int n = 10;
    vec = Vector(n, -10.0, 10.0);
    std::vector<double> v1 = MergeBatcherSeq(vec, n, 1);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Sort_Par_4thr) {
    std::vector<double> vec;
    const int n = 100;
    vec = Vector(n, -10.0, 10.0);
    std::vector<double> v1 = MergeBatcherPar(vec, n, 4);
    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(v1, vec);
}
TEST(Radix_Sort_Merge_Batcher, Test_Sort_Par_Seq) {
    std::vector<double> vec;
    const int n = 100;
    int thr = 5;
    vec = Vector(n, -10000.0, 10000.0);
    std::vector<double> vec1 = vec;
    std::vector<double> v1 = MergeBatcherPar(vec, n, thr);
    std::vector<double> v2 = MergeBatcherSeq(vec, n, thr);
    ASSERT_EQ(v1, v2);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
