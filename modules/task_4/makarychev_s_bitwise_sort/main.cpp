// Copyright 2021 Makarychev Sergey
#include <time.h>
#include <gtest/gtest.h>
#include <utility>
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
    signedRadixSortStd(&vec1);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_seq_omp) {
    std::vector<int> vec1 = { 1, 3, 7, 4, 0, 7, 0, 9 };
    std::vector<int> vec2 = vec1;
    signedRadixSortStd(&vec1);
    signedRadixSort(&vec2);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_litle_negative_value) {
    std::vector<int> vec1 = { -1, -8, -4, -10, -1, -6, -6, -6 };
    std::vector<int> vec2 = vec1;
    signedRadixSortStd(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_zero_value) {
    std::vector<int> vec1 = { 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> vec2 = vec1;
    signedRadixSortStd(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_mixed_value) {
    std::vector<int> vec1 = { 2, -99947, -9, 87456, -987456, 0 };
    std::vector<int> vec2 = vec1;
    signedRadixSortStd(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_just_sorted_vector) {
    std::vector<int> vec1 = { -99999, -1, 0, 8, 9, 15, 10265, 235458 };
    std::vector<int> vec2 = vec1;
    signedRadixSortStd(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

/*TEST(Omp_Radix_Sort, sort_vector_of_random_value_with_time) {
    time_t t1, t2;
    std::vector<int> vec1 = getRandomVector(15000000);
    std::vector<int> vec2 = vec1;
    std::cout << std::endl;
    t1 = clock();
    signedRadixSortStd(&vec1);
    t2 = clock();
    double par = t2 - t1;
    std::cout << "Par: " << par / CLOCKS_PER_SEC << std::endl;
    t1 = clock();
    signedRadixSort(&vec2);
    t2 = clock();
    double seq = t2 - t1;
    std::cout << "Seq: " << seq / CLOCKS_PER_SEC << std::endl;
    std::cout << "Acceleration: " << seq / par << std::endl;
    ASSERT_EQ(vec1, vec2);
}*/
