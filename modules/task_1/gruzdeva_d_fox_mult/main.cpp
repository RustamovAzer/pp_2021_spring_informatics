// Copyright 2021 Gruzdeva Diana

#include <gtest/gtest.h>
#include "../../../modules/task_1/gruzdeva_d_fox_mult/fox_mult.h"

TEST(Matrix_Multiplication, CHECKING_DIRECT_MULTIPLICATION) {
    std::vector<double> A = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> B = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> C = {1, 4, 10, 20,
                             0, 1, 4, 10,
                             0, 0, 1, 4,
                             0, 0, 0, 1};

    std::vector<double> C_Calculated = directMultiplication(A, B);

    ASSERT_EQ(C_Calculated, C);
}

TEST(Matrix_Multiplication, CHECKING_FOX_MULTIPLICATION) {
    std::vector<double> A = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> B = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> C = {1, 4, 10, 20,
                             0, 1, 4, 10,
                             0, 0, 1, 4,
                             0, 0, 0, 1};

    std::vector<double> C_Calculated = foxMultiplication(A, B, 2);

    ASSERT_EQ(C_Calculated, C);
}

TEST(Matrix_Multiplication, CHECKING_FOX_MULT_DIFFERENT_BLOCK_SIZE) {
    std::vector<double> A = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> B = {1, 2, 3, 4,
                             0, 1, 2, 3,
                             0, 0, 1, 2,
                             0, 0, 0, 1};

    std::vector<double> C = {1, 4, 10, 20,
                             0, 1, 4, 10,
                             0, 0, 1, 4,
                             0, 0, 0, 1};

    std::vector<double> C_Calculated = foxMultiplication(A, B, 4);

    ASSERT_EQ(C_Calculated, C);
}

TEST(Matrix_Multiplication, EQUALITY_OF_DIRECT_AND_FOX_METHODS) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    std::vector<double> C1 = directMultiplication(A, B);

    std::vector<double> C2 = foxMultiplication(A, B, 2);

    ASSERT_EQ(C1, C2);
}

TEST(Matrix_Multiplication, FOX_METHOD_WITH_DIFF_BLOCK_SIZE) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    std::vector<double> C1 = foxMultiplication(A, B, 4);

    std::vector<double> C2 = foxMultiplication(A, B, 2);

    ASSERT_EQ(C1, C2);
}

TEST(Matrix_Multiplication, FOX_METHOD_WITH_DIFF_BLOCK_SIZE_LARGER) {
    std::vector<double> A = getRandomMatrix(100, time(0));

    std::vector<double> B = getRandomMatrix(100, time(0) + 1);

    std::vector<double> C1 = foxMultiplication(A, B, 50);

    std::vector<double> C2 = foxMultiplication(A, B, 10);

    ASSERT_EQ(C1, C2);
}

TEST(Matrix_Multiplication, THROW_DIFFERENT_MATRICES_SIZES) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(5, time(0) + 1);

    ASSERT_ANY_THROW(foxMultiplication(A, B, 1));
}

TEST(Matrix_Multiplication, THROW_LARGE_BLOCK_SIZE) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(foxMultiplication(A, B, 17));
}

TEST(Matrix_Multiplication, THROW_NEGATIVE_BLOCK_SIZE) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(foxMultiplication(A, B, -1));
}

TEST(Matrix_Multiplication, THROW_WRONG_BLOCK_SIZE) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(foxMultiplication(A, B, 5));
}

TEST(Matrix_Multiplication, THROW_NULL_BLOCK_SIZE) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(foxMultiplication(A, B, 0));
}
