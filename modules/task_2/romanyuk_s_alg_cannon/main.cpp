// Copyright 2021 Romanyuk Sergey
#include <gtest/gtest.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include "../../../modules/task_2/romanyuk_s_alg_cannon/cannon.h"

TEST(Cannon_Multiplication, Test1_NegativeSize) {
    std::vector<double> A = genMatrix(4);

    std::vector<double> B = genMatrix(4);

    ASSERT_ANY_THROW(CannonMultiplication(A, B, -1));
}

TEST(Cannon_Multiplication, Test2_LargeBlockSize) {
    std::vector<double> A = genMatrix(4);

    std::vector<double> B = genMatrix(4);

    ASSERT_ANY_THROW(CannonMultiplication(A, B, 10));
}

TEST(Cannon_Multiplication, Test3_Time_Parallel) {
    std::vector<double> A = genMatrix(4);
    std::vector<double> B = genMatrix(4);

    double t1 = omp_get_wtime();
    std::vector<double> C1 = CannonMultiplication(A, B, 4);
    t1 = omp_get_wtime() - t1;

    double t2 = omp_get_wtime();
    std::vector<double> C2 = parallelCannonMult(A, B, 4);
    t2 = omp_get_wtime() - t2;

    ASSERT_TRUE(DifferenceMatrix(C1, C2));
}

TEST(Cannon_Multiplication, Test4_Result_Algorithms) {
    std::vector<double> A = genMatrix(4);
    std::vector<double> B = genMatrix(4);

    ASSERT_TRUE(DifferenceMatrix(SequentinalMultiMatrix(A, B, 4),
        parallelCannonMult(A, B, 4)));
}

TEST(Cannon_Multiplication, Test5_Cannon_Parall_Multiplication) {
    std::vector<double> A = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> B = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> C = { 1, 2, 5, 8,
                             0, 1, 2, 5,
                             0, 0, 1, 2,
                             0, 0, 0, 1 };

    std::vector<double> MultiMatrix = CannonMultiplication(A, B, 4);
    ASSERT_TRUE(DifferenceMatrix(C, MultiMatrix));
}
