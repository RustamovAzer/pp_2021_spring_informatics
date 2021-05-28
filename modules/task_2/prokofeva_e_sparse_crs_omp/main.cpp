// Copyright 2021 Prokofeva Elizaveta
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "../../../modules/task_2/prokofeva_e_sparse_crs_omp/sparse_crs.h"

TEST(Sparse_crs_matrix, Test1) {
    crs_matrix first = generate(50);
    crs_matrix second = generate(50);
    crs_matrix res = mult(first, second);
    crs_matrix resp = parallel_mult(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test2) {
    crs_matrix first = generate(100);
    crs_matrix second = generate(100);
    crs_matrix res = mult(first, second);
    crs_matrix resp = parallel_mult(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test3) {
    crs_matrix first = generate(200);
    crs_matrix second = generate(200);
    crs_matrix res = mult(first, second);
    crs_matrix resp = parallel_mult(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test4) {
    crs_matrix first = generate(350);
    crs_matrix second = generate(350);
    crs_matrix res = mult(first, second);
    crs_matrix resp = parallel_mult(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test5) {
    crs_matrix first = generate(500);
    crs_matrix second = generate(500);
    crs_matrix res = mult(first, second);
    crs_matrix resp = parallel_mult(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
