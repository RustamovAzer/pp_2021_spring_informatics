// Copyright 2021 Makarychev Sergey
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "./bitwise_sort.h"

TEST(Sequential_Radix_Sort, sort_vector_of_random_value) {
    std::vector<int> vec1 = getRandomVector(78);
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSort(&vec1);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_litle_positive_value) {
    std::vector<int> vec1 = {1, 3, 7, 4, 0, 7, 0, 9};
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSortOmp(&vec1);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_seq_omp) {
    std::vector<int> vec1 = { 1, 3, 7, 4, 0, 7, 0, 9 };
    std::vector<int> vec2 = vec1;
    signedRadixSortOmp(&vec1);
    signedRadixSort(&vec2);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_litle_negative_value) {
    std::vector<int> vec1 = { -1, -8, -4, -10, -1, -6, -6, -6 };
    std::vector<int> vec2 = vec1;
    signedRadixSortOmp(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_zero_value) {
    std::vector<int> vec1 = { 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> vec2 = vec1;
    signedRadixSortOmp(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_mixed_value) {
    std::vector<int> vec1 = { 2, -99947, -9, 87456, -987456, 0 };
    std::vector<int> vec2 = vec1;
    signedRadixSortOmp(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_just_sorted_vector) {
    std::vector<int> vec1 = { -99999, -1, 0, 8, 9, 15, 10265, 235458 };
    std::vector<int> vec2 = vec1;
    signedRadixSortOmp(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

/*TEST(Omp_Radix_Sort, sort_vector_of_random_value_with_time) {
    std::vector<int> vec1 = getRandomVector(150000000);
    std::vector<int> vec2 = vec1;
    std::cout << std::endl;
    double t1 = omp_get_wtime();
    signedRadixSortOmp(&vec1);
    double t2 = omp_get_wtime();
    double ompTime = t2 - t1;
    std::cout << "Par: " << ompTime << std::endl;
    t1 = omp_get_wtime();
    signedRadixSort(&vec2);
    t2 = omp_get_wtime();
    double seqTime = t2 - t1;
    std::cout << "Seq: " << seqTime << std::endl;
    std::cout << "Acceleration: " << seqTime / ompTime << std::endl;
    ASSERT_EQ(vec1, vec2);
}*/
