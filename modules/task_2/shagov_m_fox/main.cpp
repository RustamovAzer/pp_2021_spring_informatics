// Copyright 2021 Shagov Maksim
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/shagov_m_fox/fox.h"

TEST(Shagov_Maksim_Omp, Test_Create_Random_Matrix) {
    const int count = 10;
    Matrix A = createRandomMatrix(count);
    Matrix B(A);
    ASSERT_EQ(A, B);
}

TEST(Shagov_Maksim_Omp, Test_Throw_Exeption_Dif_Size) {
    Matrix A = {1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1};
    Matrix B(1);
    ASSERT_ANY_THROW(sequentialBlockMatrixMultiplication(A, B, 4 * 4));
}

TEST(Shagov_Maksim_Omp, Test_Throw_Exeption_Null_Block_Size) {
    Matrix A;
    Matrix B;
    ASSERT_ANY_THROW(sequentialBlockMatrixMultiplication(A, B, 0));
}

TEST(Shagov_Maksim_Omp, Test_Throw_Exeption_Null_Matrix_Size) {
    Matrix A;
    Matrix B;
    ASSERT_ANY_THROW(sequentialBlockMatrixMultiplication(A, B, 10));
}

TEST(Shagov_Maksim_Omp, Test_Throw_Exeption_No_Square_Matrix_Size) {
    Matrix A(1, 0);
    Matrix B(1, 0);
    ASSERT_ANY_THROW(sequentialBlockMatrixMultiplication(A, B, 13));
}

TEST(Shagov_Maksim_Omp, Test_Identity_Matrix_Mult) {
    Matrix A = {1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1};
    Matrix B(A);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    ASSERT_EQ(A, C);
}

TEST(Shagov_Maksim_Omp, Test_Top_Matrix_Mult) {
    Matrix A = {1, 2, 3, 4,
                0, 2, 3, 4,
                0, 0, 3, 4,
                0, 0, 0, 4};
    Matrix B(A);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    Matrix C_my_result = {1, 6, 18, 40,
                          0, 4, 15, 36,
                          0, 0, 9,  28,
                          0, 0, 0,  16};
    ASSERT_EQ(C_my_result, C);
}

TEST(Shagov_Maksim_Omp, Test_4_on_4_Some_Matrix_Mult) {
    Matrix A = {1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4};
    Matrix B(A);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    Matrix C_block = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    ASSERT_EQ(C, C_block);
}

TEST(Shagov_Maksim_Omp, Test_4_on_4_Reverse_Matrix_Mult) {
    Matrix A = {1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4};
    Matrix B = {4, 3, 2, 1,
                4, 3, 2, 1,
                4, 3, 2, 1,
                4, 3, 2, 1};
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    Matrix C_my_result = {40, 30, 20, 10,
                          40, 30, 20, 10,
                          40, 30, 20, 10,
                          40, 30, 20, 10};
    ASSERT_EQ(C_my_result, C);
}

TEST(Shagov_Maksim_Omp, Test_4_on_4_Matrix_Mult) {
    Matrix A = {4,    18,    7,   -2,
                0,    12,    1.3,  -5.7,
                0.01, 1.2,  -1.2, -4,
                0.3,  0,     7,   -2.5};
    Matrix B = {4.3,  -150,   14.14, 5.1,
                1,   -1,      0,    -2,
                2,    15.6,   6.15,  14.88,
                88.14, 2.5,  -7.3,   5};
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 4 * 4);
    Matrix C_my_result = {-127.08,  -513.8, 114.21,   78.56,
                          -487.798, -5.97,  49.605,  -33.156,
                          -353.717, -31.42, 21.9614, -40.205,
                          -205.06,   57.95, 65.542,   93.19};
    ASSERT_TRUE(isEqualMatrix(C_my_result, C));
}

TEST(Shagov_Maksim_Omp, Test_1_on_1_Some_Matrix_Mult) {
    Matrix A = {1};
    Matrix B(A);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 1 * 1);
    ASSERT_EQ(A, C);
}

TEST(Shagov_Maksim_Omp, Test_1_on_1_Dif_Matrix_Mult) {
    Matrix A = {1};
    Matrix B(A);
    B[0] = 3;
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 1 * 1);
    ASSERT_EQ(B, C);
}

TEST(Shagov_Maksim_Omp, Test_8_on_8_Some_Matrix_Mult) {
    Matrix A = {1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8};
    Matrix B(A);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 8 * 8);
    Matrix C_my_result = {36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288,
                          36, 72, 108, 144, 180, 216, 252, 288};
    ASSERT_EQ(C_my_result, C);
}

TEST(Shagov_Maksim_Omp, Test_8_on_8_Identity_on_Random_Matrix_Mult) {
    Matrix A = {1, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1};
    Matrix B = createRandomMatrix(8 * 8);
    Matrix C = sequentialBlockMatrixMultiplication(A, B, 8 * 8);
    ASSERT_EQ(B, C);
}

TEST(Shagov_Maksim_Omp, Test_4_on_4_Random_Matrix_Mult) {
    size_t size = 4;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    int t_count = 4;
    omp_set_num_threads(t_count);
    Matrix C = parallelBlockMatrixMultiplication(A, B, size * size);
    t_count = 1;
    omp_set_num_threads(t_count);
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    ASSERT_TRUE(isEqualMatrix(C, C_block));
}

TEST(Shagov_Maksim_Omp, Test_8_on_8_Random_Matrix_Mult) {
    size_t size = 8;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    int t_count = 4;
    omp_set_num_threads(t_count);
    Matrix C = parallelBlockMatrixMultiplication(A, B, size * size);
    t_count = 1;
    omp_set_num_threads(t_count);
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    ASSERT_TRUE(isEqualMatrix(C, C_block));
}

TEST(Shagov_Maksim_Omp, Test_50_on_50_Random_Matrix_Mult) {
    size_t size = 50;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    int t_count = 4;
    omp_set_num_threads(t_count);
    double t1 = omp_get_wtime();
    Matrix C = parallelBlockMatrixMultiplication(A, B, size * size);
    double t2 = omp_get_wtime();
    std::cout << "Omp: " << t2 - t1 << std::endl;
    t_count = 1;
    omp_set_num_threads(t_count);
    double t3 = omp_get_wtime();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    double t4 = omp_get_wtime();
    std::cout << "Seq: " << t4 - t3 << std::endl;
    std::cout << (t4 - t3) / (t2 - t1);
    ASSERT_TRUE(isEqualMatrix(C, C_block));
}
