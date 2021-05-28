// Copyright 2021 Tyurmina Alexandra

#include <gtest/gtest.h>
#include <vector>
#include "./shell_betcher.h"

TEST(shell_betcher, Shell) {
    std::vector<int> vec = { 3, -1, 12, 8, 5 };
    std::vector<int> res = ShellSort(vec, 5);
    std::vector<int> correct_res = { -1, 3, 5, 8, 12 };

    ASSERT_EQ(res, correct_res);
}

TEST(shell_betcher, BetcherMerge) {
    std::vector<int> vec1 = { 15, 47, 5, 91 };
    std::vector<int> vec2 = { 62, 35, 78, 94 };
    std::vector<int> correct_res = { 15, 47, 35, 5, 62, 78, 91, 94 };
    std::vector<int> res = BetcherMerge(vec1, vec2);

    ASSERT_EQ(res, correct_res);
}

TEST(shell_betcher, shell_b_10) {
    std::vector<int> vec = { 15, 47, 5, 91, 82, 30, 12, 64, 70, 13 };
    std::vector<int> sort = ShellBetcherS(vec, 10);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(vec, sort);
}

TEST(shell_betcher, shell_b_15) {
    std::vector<int> vec = { 15, 47, 5, 91, 82, 30, 12, 64, 70, 13, 78, 54, 35, 62, 24 };
    std::vector<int> sort = ShellBetcherS(vec, 15);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(vec, sort);
}

TEST(shell_betcher, shell_b_20) {
    std::vector<int> vec = { 15, 47, 5, 91, 82, 30, 12, 64, 70, 13, 78, 54, 35, 62, 24, 55, 9, 28, 105, 40 };
    std::vector<int> sort = ShellBetcherS(vec, 20);
    std::sort(vec.begin(), vec.end());

    ASSERT_EQ(vec, sort);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
