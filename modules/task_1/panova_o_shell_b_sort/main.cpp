// Copyright 2021 Panova Olga
#include <gtest/gtest.h>
#include <vector>
#include "./shell_b_sort.h"
TEST(Odd_Even_Shell, ShellSort_Is_Correct) {
    std::vector<int> a = { 10, 9, 18, 7, 6, -5, 1, 22 };
    a = ShellSort(a);
    std::vector<int> b = { -5, 1, 6, 7, 9, 10, 18, 22 };
    ASSERT_EQ(a, b);
}
TEST(Odd_Even_Shell, BatcherMerge_Is_Correct) {
    std::vector<int> a = { 38, 40, 28, 22, 94, 12 };
    std::vector<int> b = { 29, 98, 9, 67, 14, 13 };
    std::vector<int> expected_res = { 29, 38, 40, 28, 9, 22, 67, 94, 14, 12, 13, 98 };
    std::vector<int> res = BatcherMerge(a, b);
    ASSERT_EQ(expected_res, res);
}
TEST(Odd_Even_Shell, Vec_1_Sort) {
    std::vector<int> a = CreateVector(1);
    std::vector<int> b = OddEvenShellSort(a);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
TEST(Odd_Even_Shell, Vec_20_Sort) {
    std::vector<int> a = CreateVector(20);
    std::vector<int> b = OddEvenShellSort(a);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
TEST(Odd_Even_Shell, Vec_10000_Sort) {
    std::vector<int> a = CreateVector(10000);
    std::vector<int> b = OddEvenShellSort(a);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
