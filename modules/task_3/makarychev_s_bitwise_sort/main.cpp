// Copyright 2021 Makarychev Sergey
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
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
    std::vector<int> vec1 = { 1, 3, 7, 4, 0, 7, 0, 9 };
    std::vector<int> vec2 = vec1;
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    signedRadixSortTbb(&vec1);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_seq_omp) {
    std::vector<int> vec1 = { 1, 3, 7, 4, 0, 7, 0, 9 };
    std::vector<int> vec2 = vec1;
    signedRadixSortTbb(&vec1);
    signedRadixSort(&vec2);
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_litle_negative_value) {
    std::vector<int> vec1 = { -1, -8, -4, -10, -1, -6, -6, -6 };
    std::vector<int> vec2 = vec1;
    signedRadixSortTbb(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_zero_value) {
    std::vector<int> vec1 = { 0, 0, 0, 0, 0, 0, 0, 0 };
    std::vector<int> vec2 = vec1;
    signedRadixSortTbb(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_vector_of_mixed_value) {
    std::vector<int> vec1 = { 2, -99947, -9, 87456, -987456, 0 };
    std::vector<int> vec2 = vec1;
    signedRadixSortTbb(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

TEST(Omp_Radix_Sort, sort_just_sorted_vector) {
    std::vector<int> vec1 = { -99999, -1, 0, 8, 9, 15, 10265, 235458 };
    std::vector<int> vec2 = vec1;
    signedRadixSortTbb(&vec1);
    std::sort(vec2.begin(), vec2.begin() + vec2.size());
    ASSERT_EQ(vec1, vec2);
}

/* TEST(Seqvssspar, Test_Only_Positive) {
     std::vector<int> vec1 = getRandomVector(15000000);
     std::vector<int> vec2 = vec1;
     tbb::tick_count t1Seq = tbb::tick_count::now();
     signedRadixSort(&vec1);
     tbb::tick_count t2Seq = tbb::tick_count::now();
     double time_seq = (t2Seq - t1Seq).seconds();
     std::cout << "Seq: " << time_seq << std::endl;
     tbb::tick_count t1_par = tbb::tick_count::now();
     signedRadixSortTbb(&vec2);
     tbb::tick_count t2_par = tbb::tick_count::now();
     double time_par = (t2_par - t1_par).seconds();
     std::cout << "Par: " << time_par << std::endl;
     std::cout << "Acceleration " << time_seq / time_par << std::endl;
     ASSERT_EQ(vec2, vec1);
 }*/
