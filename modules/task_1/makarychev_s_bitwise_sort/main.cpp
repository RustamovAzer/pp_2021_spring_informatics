// Copyright 2021 Makarychev Sergey
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "./bitwise_sort.h"

TEST(Sequential_Radix_Sort, sort_vector_of_random_value_1) {
    std::vector<int> vec1 = getRandomVector(50);
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSort(vec1.data(), vec1.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_vector_of_random_value_2) {
    std::vector<int> vec1 = getRandomVector(78);
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSort(vec1.data(), vec1.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_vector_of_litle_positive_value) {
    std::vector<int> vec1 = {1, 3, 7, 4, 0, 7, 0, 9};
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSort(vec1.data(), vec1.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_vector_of_litle_negative_value) {
    std::vector<int> vec1 = { -1, -8, -4, -10, -1, -6, -6, -6 };
    std::vector<int> vec2 = vec1;
    signedRadixSort(vec1.data(), vec1.size());
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_vector_of_zero_value) {
    std::vector<int> vec1 = { 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> vec2 = vec1;
    signedRadixSort(vec1.data(), vec1.size());
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_vector_of_mixed_value) {
    std::vector<int> vec1 = { 2, -99947, -9, 87456, -987456, 0 };
    std::vector<int> vec2 = vec1;
    signedRadixSort(vec1.data(), vec1.size());
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Sequential_Radix_Sort, sort_just_sorted_vector) {
    std::vector<int> vec1 = { -99999, -1, 0, 8, 9, 15, 10265, 235458 };
    std::vector<int> vec2 = vec1;
    signedRadixSort(vec1.data(), vec1.size());
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}
