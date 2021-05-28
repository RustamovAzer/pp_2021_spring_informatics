// Copyright 2021 Romanuyk Sergey
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include"../../../modules/task_1/romanuyk_algoritm_kennona/algoritm_kennona.h"

TEST(Matrix_Multiplication, Test1_SequentinalMultiMatrix) {
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

    std::vector<double> MultiMatrix = SequentinalMultiMatrix(A, B, 4);
    ASSERT_EQ(MultiMatrix, C);
}

TEST(Matrix_Multiplication, Test2_KennonMultiplication) {
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

    std::vector<double> MultiMatrix = KennonMultiplication(A, B, 2);
    ASSERT_EQ(MultiMatrix, C);
}

TEST(Matrix_Multiplication, Test3_NegativeSize) {
    std::vector<double> A = genMatrix(4);

    std::vector<double> B = genMatrix(4);

    ASSERT_ANY_THROW(KennonMultiplication(A, B, -1));
}

TEST(Matrix_Multiplication, Test3_LargeBlockSize) {
    std::vector<double> A = genMatrix(4);

    std::vector<double> B = genMatrix(4);

    ASSERT_ANY_THROW(KennonMultiplication(A, B, 10));
}

TEST(Matrix_Multiplication, Text4_Sequentinal_and_KennonMultiplication) {
    std::vector<double> A = genMatrix(4);

    std::vector<double> B = genMatrix(4);

    std::vector<double> C1 = SequentinalMultiMatrix(A, B, 4);

    std::vector<double> C2 = KennonMultiplication(A, B, 2);

    ASSERT_EQ(C1, C2);
}
