// Copyright 2021 Alibekov Murad
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <iostream>
#include <ctime>
#include "./ccs_complex_matrix.h"

TEST(SPARSE_MATRICES, THROWS_WHEN_N_IS_NEGATIVE) {
    ASSERT_ANY_THROW(generate_regular_ccs(10728, -100, 3));
}

TEST(SPARSE_MATRICES, THROWS_WHEN_N_IS_ZERO) {
    ASSERT_ANY_THROW(generate_regular_ccs(10728, 0, 3));
}

TEST(SPARSE_MATRICES, TEST_EQUAL_MATRICES) {
    ccs_complex_matrix sparse_matrix(4, 6);
    sparse_matrix.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_2(4, 6);
    sparse_matrix_2.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix_2.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix_2.col_indexes = { 0, 1, 2, 4, 6 };

    ASSERT_TRUE(sparse_matrix_2 == sparse_matrix);
}

TEST(SPARSE_MATRICES, TEST_TRANSPOSED_SPARSE_MATRIX) {
    ccs_complex_matrix sparse_matrix(4, 6);
    sparse_matrix.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_T(4, 6);
    sparse_matrix_T.values = { 3, 7, 8, 9, 15, 16 };
    sparse_matrix_T.rows = { 1, 3, 2, 0, 2, 3 };
    sparse_matrix_T.col_indexes = { 0, 2, 3, 3, 6 };

    EXPECT_EQ(sparse_matrix_T, transpose(sparse_matrix));
}

TEST(SPARSE_MATRICES, TEST_DOUBLE_TRANSPOSED_SPARSE_MATRIX) {
    ccs_complex_matrix sparse_matrix(4, 6);
    sparse_matrix.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix.col_indexes = { 0, 1, 2, 4, 6 };

    EXPECT_EQ(sparse_matrix, transpose(transpose(sparse_matrix)));
}

TEST(SPARSE_MATRICES, PRINT_SPARSE_MATRIX) {
    ccs_complex_matrix sparse_matrix(4, 6);
    sparse_matrix.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix.col_indexes = { 0, 1, 2, 4, 6 };

    std::cout << "\tFirst type:\n";
    PrintCCSMatrix(sparse_matrix);
    std::cout << std::endl;

    std::cout << "\tSecond type:\n";
    PrintCCSMatrix(sparse_matrix, false);
    std::cout << std::endl;

    std::cout << "\tThird type:\n";
    PrintDensificationOfCCSMatrix(sparse_matrix);
    std::cout << std::endl;

    std::cout << "\tFourth type:\n";
    PrintDensificationOfCCSMatrix(sparse_matrix, false);
    std::cout << std::endl;
}

TEST(SPARSE_MATRICES, PERFORMANCE_MEASUREMENT_OF_GENERATING_BIG_SPARSE_MATRIX) {
    clock_t start_time = clock();
    EXPECT_NO_THROW(generate_regular_ccs(86538, 10000, 150));
    clock_t finish_time = clock();

    printf("\tTime  = %f\n", static_cast<double>(finish_time - start_time) / CLOCKS_PER_SEC);
}

TEST(SPARSE_MATRICES, PERFORMANCE_MEASUREMENT_OF_TRANSPOSITION_BIG_SPARSE_MATRIX) {
    ccs_complex_matrix big_sparse_matrix = generate_regular_ccs(86538, 10000, 150);

    clock_t start_time = clock();
    EXPECT_NO_THROW(transpose(big_sparse_matrix));
    clock_t finish_time = clock();

    printf("\tTime  = %f\n", static_cast<double>(finish_time - start_time) / CLOCKS_PER_SEC);
}

/////////////////////////////////////////////
///    NAIVE_MULTIPLY_SPARSE_MATRICES    ////
/////////////////////////////////////////////

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, MULTIPLY_NUMBERS) {
    ccs_complex_matrix number_1(1, 1);
    number_1.values = { 6 };
    number_1.rows = { 0 };
    number_1.col_indexes = { 0, 1 };

    ccs_complex_matrix number_2(1, 1);
    number_2.values = { 7 };
    number_2.rows = { 0 };
    number_2.col_indexes = { 0, 1 };

    ccs_complex_matrix result(1, 1);
    result.values = { 42 };
    result.rows = { 0 };
    result.col_indexes = { 0, 1 };

    EXPECT_EQ(naive_multiplicate(number_1, number_2), result);
}

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, MULTIPLY_COMPLEX_NUMBERS) {
    ccs_complex_matrix number_1(1, 1);
    number_1.values = { {-1, 2} };
    number_1.rows = { 0 };
    number_1.col_indexes = { 0, 1 };

    ccs_complex_matrix number_2(1, 1);
    number_2.values = { {3, 4} };
    number_2.rows = { 0 };
    number_2.col_indexes = { 0, 1 };

    ccs_complex_matrix result(1, 1);
    result.values = { {-11, 2} };
    result.rows = { 0 };
    result.col_indexes = { 0, 1 };

    EXPECT_EQ(naive_multiplicate(number_1, number_2), result);
}

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, THROWS_WHEN_MULTIPLY_WITH_DIFFERENT_N) {
    ccs_complex_matrix sparse_matrix_1(4, 6);
    sparse_matrix_1.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix_1.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix_1.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_2(3, 3);
    sparse_matrix_2.values = { 1, 3, 7 };
    sparse_matrix_2.rows = { 2, 2, 1 };
    sparse_matrix_2.col_indexes = { 0, 1, 2, 3 };

    ASSERT_ANY_THROW(naive_multiplicate(sparse_matrix_1, sparse_matrix_2));
}

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, MULTIPLY_SPARSE_MATRIX_AND_ZERO_MATRIX) {
    ccs_complex_matrix sparse_matrix_1(4, 6);
    sparse_matrix_1.values = { {7, 1}, {6, 4}, 2.978, {11.02, -0.9}, {-9.3, 0}, 16 };
    sparse_matrix_1.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix_1.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_2(4, 0);

    ccs_complex_matrix result(4, 0);

    EXPECT_EQ(naive_multiplicate(sparse_matrix_1, sparse_matrix_2), result);
}

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, MULTIPLY_SPARSE_MATRICES) {
    ccs_complex_matrix sparse_matrix_1(3, 4);
    sparse_matrix_1.values = { {-1, 1}, {0, 3}, 3, 0.7 };
    sparse_matrix_1.rows = { 2, 0, 2, 1 };
    sparse_matrix_1.col_indexes = { 0, 1, 3, 4 };

    ccs_complex_matrix sparse_matrix_2(3, 3);
    sparse_matrix_2.values = { {0, -3}, {0, 1}, 4 };
    sparse_matrix_2.rows = { 1, 0, 2 };
    sparse_matrix_2.col_indexes = { 0, 0, 1, 3};

    ccs_complex_matrix result(3, 4);
    result.values = { 9, {0, -9}, 2.8, {-1, -1} };
    result.rows = { 0, 2, 1, 2 };
    result.col_indexes = { 0, 0, 2, 4};
    EXPECT_EQ(naive_multiplicate(sparse_matrix_1, sparse_matrix_2), result);
}

TEST(NAIVE_MULTIPLY_SPARSE_MATRICES, PERFORMANCE_MEASUREMENT_OF_MULTIPLICATION_BIG_SPARSE_MATRICES) {
    ccs_complex_matrix big_sparse_matrix_1 = generate_regular_ccs(86538, 1000, 20);
    std::cout << "\tFirst matrix is generated!\n";

    ccs_complex_matrix big_sparse_matrix_2 = generate_regular_ccs(2395, 1000, 20);
    std::cout << "\tSecond matrix is generated!\n";

    clock_t start_time = clock();
    EXPECT_NO_THROW(naive_multiplicate(big_sparse_matrix_1, big_sparse_matrix_2));
    clock_t finish_time = clock();

    printf("\tTime  = %f\n", static_cast<double>(finish_time - start_time) / CLOCKS_PER_SEC);
}

/////////////////////////////////////////////
///    OPTIM_MULTIPLY_SPARSE_MATRICES    ////
/////////////////////////////////////////////

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, MULTIPLY_NUMBERS) {
    ccs_complex_matrix number_1(1, 1);
    number_1.values = { 6 };
    number_1.rows = { 0 };
    number_1.col_indexes = { 0, 1 };

    ccs_complex_matrix number_2(1, 1);
    number_2.values = { 7 };
    number_2.rows = { 0 };
    number_2.col_indexes = { 0, 1 };

    ccs_complex_matrix result(1, 1);
    result.values = { 42 };
    result.rows = { 0 };
    result.col_indexes = { 0, 1 };

    EXPECT_EQ(optim_multiplicate(number_1, number_2), result);
}

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, MULTIPLY_COMPLEX_NUMBERS) {
    ccs_complex_matrix number_1(1, 1);
    number_1.values = { {-1, 2} };
    number_1.rows = { 0 };
    number_1.col_indexes = { 0, 1 };

    ccs_complex_matrix number_2(1, 1);
    number_2.values = { {3, 4} };
    number_2.rows = { 0 };
    number_2.col_indexes = { 0, 1 };

    ccs_complex_matrix result(1, 1);
    result.values = { {-11, 2} };
    result.rows = { 0 };
    result.col_indexes = { 0, 1 };

    EXPECT_EQ(optim_multiplicate(number_1, number_2), result);
}

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, THROWS_WHEN_MULTIPLY_WITH_DIFFERENT_N) {
    ccs_complex_matrix sparse_matrix_1(4, 6);
    sparse_matrix_1.values = { 9, 3, 8, 15, 7, 16 };
    sparse_matrix_1.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix_1.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_2(3, 3);
    sparse_matrix_2.values = { 1, 3, 7 };
    sparse_matrix_2.rows = { 2, 2, 1 };
    sparse_matrix_2.col_indexes = { 0, 1, 2, 3 };

    ASSERT_ANY_THROW(optim_multiplicate(sparse_matrix_1, sparse_matrix_2));
}

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, MULTIPLY_SPARSE_MATRIX_AND_ZERO_MATRIX) {
    ccs_complex_matrix sparse_matrix_1(4, 6);
    sparse_matrix_1.values = { {7, 1}, {6, 4}, 2.978, {11.02, -0.9}, {-9.3, 0}, 16 };
    sparse_matrix_1.rows = { 3, 0, 1, 3, 0, 3 };
    sparse_matrix_1.col_indexes = { 0, 1, 2, 4, 6 };

    ccs_complex_matrix sparse_matrix_2(4, 0);

    ccs_complex_matrix result(4, 0);

    EXPECT_EQ(optim_multiplicate(sparse_matrix_1, sparse_matrix_2), result);
}

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, MULTIPLY_SPARSE_MATRICES) {
    ccs_complex_matrix sparse_matrix_1(3, 4);
    sparse_matrix_1.values = { {-1, 1}, {0, 3}, 3, 0.7 };
    sparse_matrix_1.rows = { 2, 0, 2, 1 };
    sparse_matrix_1.col_indexes = { 0, 1, 3, 4 };

    ccs_complex_matrix sparse_matrix_2(3, 3);
    sparse_matrix_2.values = { {0, -3}, {0, 1}, 4 };
    sparse_matrix_2.rows = { 1, 0, 2 };
    sparse_matrix_2.col_indexes = { 0, 0, 1, 3};

    ccs_complex_matrix result(3, 4);
    result.values = { 9, {0, -9}, 2.8, {-1, -1} };
    result.rows = { 0, 2, 1, 2 };
    result.col_indexes = { 0, 0, 2, 4};

    EXPECT_EQ(optim_multiplicate(sparse_matrix_1, sparse_matrix_2), result);
}

TEST(OPTIM_MULTIPLY_SPARSE_MATRICES, PERFORMANCE_MEASUREMENT_OF_MULTIPLICATION_BIG_SPARSE_MATRICES) {
    ccs_complex_matrix big_sparse_matrix_1 = generate_regular_ccs(86538, 1000, 20);
    std::cout << "\tFirst matrix is generated!\n";

    ccs_complex_matrix big_sparse_matrix_2 = generate_regular_ccs(2395, 1000, 20);
    std::cout << "\tSecond matrix is generated!\n";

    clock_t start_time = clock();
    EXPECT_NO_THROW(optim_multiplicate(big_sparse_matrix_1, big_sparse_matrix_2));
    clock_t finish_time = clock();

    printf("\tTime  = %f\n", static_cast<double>(finish_time - start_time) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
