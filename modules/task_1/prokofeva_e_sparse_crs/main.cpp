// Copyright 2021 Prokofeva Elizaveta
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "../../../modules/task_1/prokofeva_e_sparse_crs/sparse_crs.h"


TEST(Sparse_crs_matrix_1, Create) {
    std::vector<double> f = { 0, 3, 0, 7,
                              0, 0, 8, 0,
                              0, 0, 0, 0,
                              9, 0, 15, 16};
    crs_matrix first = create(4, f);
    std::vector<double> value = { 3, 7, 8, 9, 15, 16 };
    std::vector<int> col = { 1, 3, 2, 0, 2, 3 };
    std::vector<int> rowIndex = { 0, 2, 3, 3, 6 };
    ASSERT_EQ(value, first.values);
    ASSERT_EQ(col, first.cols);
    ASSERT_EQ(rowIndex, first.row_index);
}
TEST(Sparse_crs_matrix_2, Transposition) {
    std::vector<double> f = { 0, 3, 0, 7,
                              0, 0, 8, 0,
                              0, 0, 0, 0,
                              9, 0, 15, 16 };
    crs_matrix first = create(4, f);
    first = transpose(first);
    std::vector<double> value = { 9, 3, 8, 15, 7, 16 };
    std::vector<int> col = { 3, 0, 1, 3, 0, 3 };
    std::vector<int> rowIndex = { 0, 1, 2, 4, 6 };
    ASSERT_EQ(value, first.values);
    ASSERT_EQ(col, first.cols);
    ASSERT_EQ(rowIndex, first.row_index);
}
TEST(Sparse_crs_matrix_3, Multiplication_3x3) {
    std::vector<double> f = { 0, 2, 0,
                             10, 0, 0,
                              0, 0, 6};
    crs_matrix first = create(3, f);
    std::vector<double> s = { 1, 0, 0,
                              0, 5, 0,
                              0, 14, 3};
    crs_matrix second = create(3, s);
    crs_matrix result = mult(first, second);
    std::vector<double> r = { 10, 10, 84, 18 };
    ASSERT_EQ(r, result.values);
}
TEST(Sparse_crs_matrix_4, Multiplication_3x3) {
    std::vector<double> f = { 0, 0.5, 1.5,
                              2, 0, 0,
                              0, 6.5, 0};
    crs_matrix first = create(3, f);
    std::vector<double> s = { 3.5, 0, 0,
                               0, 1, 4.5,
                               0, 0, 0};
    crs_matrix second = create(3, s);
    crs_matrix result = mult(first, second);
    std::vector<double> r = { 0.5, 2.25, 7, 6.5, 29.25 };
    ASSERT_EQ(r, result.values);
}
TEST(Sparse_crs_matrix_5, Multiplication_4x4) {
    std::vector<double> f = { 0, 11, 0, 25,
                              96, 0, 0, 0,
                              0, 0, 15, 0,
                              0, 44, 0, 0};
    crs_matrix first = create(4, f);
    std::vector<double> s = { 57, 63, 0, 0,
                              0, 0, 41, 0,
                              0, 78, 0, 0,
                              0, 0, 0, 22};
    crs_matrix second = create(4, s);
    crs_matrix result = mult(first, second);
    std::vector<double> r = { 451, 550, 5472, 6048, 1170, 1804 };
    ASSERT_EQ(r, result.values);
}
TEST(Sparse_crs_matrix_6, Multiplication_zero) {
    std::vector<double> f = { 0, 1, 0,
                              1, 0, 0,
                              0, 0, 1 };
    crs_matrix first = create(3, f);
    std::vector<double> s = { 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0 };
    crs_matrix second = create(3, s);
    crs_matrix result = mult(first, second);
    ASSERT_EQ(result.values.empty(), true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
