// Copyright 2021 Tyurmina Alexandra
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <utility>
#include <functional>
#include <cmath>
#include <algorithm>
#include "./shell_betcher.h"

TEST(shell_betcher, Shell_5) {
    std::vector<int> vec = { 3, -1, 12, 8, 5 };
    std::vector<int> res = ShellSort(vec, 5);
    std::vector<int> correct_res = { -1, 3, 5, 8, 12 };

    ASSERT_EQ(res, correct_res);
}

TEST(shell_betcher, Shell_rand_10) {
    std::vector<int> vec = Random(10);
    std::vector<int> res = ShellSort(vec, 10);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(res, vec);
}

TEST(shell_betcher, betcher_merge_8) {
    std::vector<int> vec1 = { 15, 47, 5, 91 };
    std::vector<int> vec2 = { 62, 35, 78, 94 };
    std::vector<int> correct_res = { 15, 47, 35, 5, 62, 78, 91, 94 };
    std::vector<int> res = BetcherMerge(vec1, vec2);

    ASSERT_EQ(res, correct_res);
}

TEST(shell_betcher, shell_betcher_50_rand) {
    std::vector<int> vec = Random(50);
    std::vector<int> sort = ShellBetcher(vec, 50);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(vec, sort);
}

TEST(shell_betcher, shell_betcher_10) {
    std::vector<int> vec = { 15, 47, 5, 91, 82, 30, 12, 64, 70, 13 };
    std::vector<int> sort = ShellBetcher(vec, 10);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(vec, sort);
}

TEST(shell_betcher, shell_b_20) {
    std::vector<int> vec1 = Random(20);
    std::vector<int> vec2 = ShellBetcherOmp(vec1, 4, 20);
    std::sort(vec1.begin(), vec1.end());

    ASSERT_EQ(vec1, vec2);
}

TEST(shell_betcher, shell_b_100) {
    std::vector<int> vec1 = Random(100);
    std::vector<int> vec2 = ShellBetcherOmp(vec1, 4, 100);
    std::sort(vec1.begin(), vec1.end());

    ASSERT_EQ(vec1, vec2);
}

TEST(shell_betcher, shell_b_1000) {
    std::vector<int> vec1 = Random(1000);
    std::vector<int> vec2 = ShellBetcherOmp(vec1, 4, 1000);
    std::sort(vec1.begin(), vec1.end());

    ASSERT_EQ(vec1, vec2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
