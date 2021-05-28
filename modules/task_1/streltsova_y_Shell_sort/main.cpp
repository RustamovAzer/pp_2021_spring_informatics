// Copyright 2021 Streltsova Yana
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./Shell_sort.h"


TEST(Seq_Shell_sort, Test_zero_size) {
    const int size_vector = 0;
    std::vector<int> a = getRandomVector(size_vector);
    std::vector<int> result;
    shell_sort(a.begin(), a.end(), compare<int>);
    ASSERT_EQ(result, a);
}

TEST(Seq_Shell_sort, Test_one_element) {
    const int size_vector = 1;
    std::vector<int> a = getRandomVector(size_vector);
    std::vector<int> result(a);
    shell_sort(a.begin(), a.end(), compare<int>);
    ASSERT_EQ(result, a);
}

TEST(Seq_Shell_sort, Test_normal_size) {
    const int size_vector = 35;
    std::vector<int> a = getRandomVector(size_vector);
    std::vector<int> result(a);
    std::sort(result.begin(), result.end());
    shell_sort(a.begin(), a.end(), compare<int>);
    ASSERT_EQ(result, a);
}

TEST(Seq_Shell_sort, Test_sorted_vector) {
    const int size_vector = 100;
    std::vector<int> a = getRandomVector(size_vector);
    std::sort(a.begin(), a.end());
    std::vector<int> result(a);
    shell_sort(a.begin(), a.end(), compare<int>);
    ASSERT_EQ(result, a);
}

TEST(Seq_Shell_sort, Test_big_size) {
    const int size_vector = 1000000;
    std::vector<int> a = getRandomVector(size_vector);
    std::vector<int> result(a);
    std::sort(result.begin(), result.end());
    shell_sort(a.begin(), a.end(), compare<int>);
    ASSERT_EQ(result, a);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
