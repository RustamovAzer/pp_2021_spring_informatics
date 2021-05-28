// Copyright 2021 Prokofeva Elizaveta
#include <gtest/gtest.h>
#include <tbb/tbb.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "../../../modules/task_3/prokofeva_e_sparse_crs_tbb/sparse_crs.h"

TEST(Sparse_crs_matrix, Test1000x1000) {
    crs_matrix first = generate(1000);
    crs_matrix second = generate(1000);
    crs_matrix res = mult(first, second);
    crs_matrix resp = mult_tbb(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test2000x2000) {
    crs_matrix first = generate(2000);
    crs_matrix second = generate(2000);
    crs_matrix res = mult(first, second);
    crs_matrix resp = mult_tbb(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test3000x3000) {
    crs_matrix first = generate(3000);
    crs_matrix second = generate(3000);
    crs_matrix res = mult(first, second);
    crs_matrix resp = mult_tbb(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test4000x4000) {
    crs_matrix first = generate(4000);
    crs_matrix second = generate(4000);
    crs_matrix res = mult(first, second);
    crs_matrix resp = mult_tbb(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}
TEST(Sparse_crs_matrix, Test5000x5000) {
    crs_matrix first = generate(5000);
    crs_matrix second = generate(5000);
    crs_matrix res = mult(first, second);
    crs_matrix resp = mult_tbb(first, second);
    ASSERT_EQ(resp.values, res.values);
    ASSERT_EQ(resp.cols, res.cols);
    ASSERT_EQ(resp.row_index, res.row_index);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
