// Copyright 2021 Solovev Aleksandr
#include <numeric>
#include <utility>
#include <algorithm>
#include "gtest/gtest.h"
#include "../../../modules/task_2/solovev_a_radix_sort/radix_sort.h"


TEST(Parallel, Test_Very_Micro_Length) {
    const int length = 10;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -100, 100);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}


TEST(Parallel, Test_Very_Small_Length) {
    const int length = 100;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -1000, 1000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}

TEST(Parallel, Test_Small_length) {
    const int length = 500;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -10000, 10000);

    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}

TEST(Parallel, Test_Medium_Length) {
    const int length = 1000;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -1000, 1000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

TEST(Senquential, Test_Large_Length) {
    const int length = 1500;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -1000, 1000);
    std::vector<int> buffer2 = buffer1;

    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

TEST(Senquential, Test_Very_Large_Length) {
    const int length = 2000;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -5000, 5000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSorting(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

// TEST(Seqvssspar, Test_Only_Positive) {
//     const int length = 150000000;
//     int statusseq = 0;
//     int statuspar = 0;
//     std::vector<int> buffer1(length);
//     statusseq = generateRandomArray(&buffer1, -10000, 10000);
//     std::vector<int> buffer2 = buffer1;
//     double t1_seq = omp_get_wtime();
//     statusseq = RadixSort(&buffer1);
//     statusseq = SortingCheck(&buffer1);
//     double t2_seq = omp_get_wtime();
//     double time_seq = t2_seq - t1_seq;
//     std::cout << "Time seq: " << time_seq << std::endl;
//     double t1_omp = omp_get_wtime();
//     statuspar = ParallelSorting(&buffer2);
//     statuspar = SortingCheck(&buffer2);
//     double t2_omp = omp_get_wtime();
//     double time_omp = t2_omp - t1_omp;
//     std::cout << "Time omp: " << time_omp << std::endl;
//     std::cout << "Acceleration: " << time_seq / time_omp << std::endl;
//     ASSERT_EQ(buffer2, buffer1);
// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
