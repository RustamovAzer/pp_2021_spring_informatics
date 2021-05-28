// Copyright 2021 Taktaev Artem
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>
#include <iostream>

#include "../../../modules/task_2/taktaev_a_qsort/qsort.h"

TEST(QSort_OMP, Test_Correct_Creating_Vector) {
    int n = 1000;
    std::vector<double> vec(n);
    ASSERT_NO_THROW(vec = createRandomVector(n));
}

TEST(QSort_OMP, Test_Sequential_Sorting_Rand_Odd_Vec) {
    int n = 11;
    std::vector<double> vec = createRandomVector(n);
    std::vector<double> vec_copy = vec;
    qSortSeq(&vec, 0, n - 1);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Sequential_Sorting_Sorted_Vec) {
    int n = 15;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
    std::vector<double> vec_copy = vec;
    qSortSeq(&vec, 0, n - 1);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Sequential_Sorting_Reverse_Sorted_Vec) {
    int n = 12;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = n - i;
    }
    std::vector<double> vec_copy = vec;
    qSortSeq(&vec, 0, n - 1);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Sequential_Sorting_Eq_Elements_Vec) {
    int n = 18;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = 1;
    }
    std::vector<double> vec_copy = vec;
    qSortSeq(&vec, 0, n - 1);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Correct_Simple_Merge) {
    std::vector<double> vec = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    std::vector<double> vec_copy = vec;
    std::sort(vec_copy.begin(), vec_copy.end());
    merge(&vec[0], 5, &vec[5], 5);
    ASSERT_EQ(vec_copy, vec);
}

TEST(QSort_OMP, Test_Parallel_Sorting_Rand_Even_Vec) {
    int n = 1300;
    std::vector<double> vec = createRandomVector(n);
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    std::sort(vec_copy.begin(), vec_copy.end());

    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Parallel_Sorting_Rand_Odd_Vec) {
    int n = 1301;
    std::vector<double> vec = createRandomVector(n);
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    std::sort(vec_copy.begin(), vec_copy.end());

    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Parallel_And_Seq_Do_The_Same) {
    int n = 1000;
    std::vector<double> vec = createRandomVector(n);
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    qSortSeq(&vec_copy, 0, n - 1);

    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Parallel_Sorting_Sorted_Vec) {
    int n = 150;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Parallel_Sorting_Reverse_Sorted_Vec) {
    int n = 120;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = n - i;
    }
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Parallel_Sorting_Eq_Elements_Vec) {
    int n = 180;
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = 1;
    }
    std::vector<double> vec_copy = vec;
    qSortOmp(&vec);
    std::sort(vec_copy.begin(), vec_copy.end());
    ASSERT_EQ(vec, vec_copy);
}

TEST(QSort_OMP, Test_Creating_Vector_Exception) {
    int n = -1;
    ASSERT_ANY_THROW(std::vector<double> vec = createRandomVector(n));
}

TEST(QSort_OMP, Test_Out_Of_Bounds_Exception) {
    int n = 11;
    std::vector<double> vec = createRandomVector(n);
    ASSERT_ANY_THROW(qSortSeq(&vec, -1, n));
}

TEST(QSort_OMP, Test_Index_Overlap_Exception) {
    int n = 10;
    std::vector<double> vec = createRandomVector(n);
    ASSERT_ANY_THROW(qSortSeq(&vec, n - 1, 0));
}

/*TEST(QSort_OMP, Test_Time) {
    int n = 1300000;
    double start = 0, finish = 0, time_seq = 0, time_par = 0;
    int n_thr = 6;  // thread num
    std::vector<double> vec = createRandomVector(n);
    std::vector<double> vec_copy = vec;
    omp_set_num_threads(n_thr);
    start = omp_get_wtime();
    qSortSeq(&vec, 0, n - 1);
    finish = omp_get_wtime();
    time_seq = finish - start;
    std::cout << "Time sequential: " << time_seq << std::endl;
    start = omp_get_wtime();
    qSortOmp(&vec_copy);
    finish = omp_get_wtime();
    time_par = finish - start;
    std::cout << "Time parallel: " << time_par << std::endl;
    std::cout << "Acceleration " << time_seq / time_par << std::endl;

    ASSERT_EQ(vec, vec_copy);
}*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
