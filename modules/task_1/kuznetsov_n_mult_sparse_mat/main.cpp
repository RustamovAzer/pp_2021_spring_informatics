// Copyright 2021 Kuznetsov Nikita
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/kuznetsov_n_mult_sparse_mat/sparse_matrix.h"

TEST(SPARSE_MATRIX, MAT_EQUAL_ITSELF_1) {
  std::vector<double> arg1 = { 1, 1, 1 };
  std::vector<int> arg2 = { 0, 1, 2, 3 };
  std::vector<int> arg3 = { 0, 1, 2 };
  sMatrix A = sMatrix(arg1, arg2, arg3, 3, 3);
  ASSERT_TRUE(A == A);
}

TEST(SPARSE_MATRIX, MAT_EQUAL_ITSELF_2) {
  std::vector<double> arg1 = { 5, 8, 1, 2, 6 };
  std::vector<int> arg2 = { 0, 2, 3, 5, 5 };
  std::vector<int> arg3 = { 0, 3, 2, 1, 3 };
  sMatrix A = sMatrix(arg1, arg2, arg3, 5, 4);
  ASSERT_TRUE(A == A);
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_TRANSPOSE) {
  std::vector<double> arg1 = { 1, 7, 1, 2, 6 };
  std::vector<int> arg2 = { 0, 2, 3, 5, 5 };
  std::vector<int> arg3 = { 0, 3, 2, 1, 3 };
  sMatrix A = sMatrix(arg1, arg2, arg3, 5, 4);
  ASSERT_NO_THROW(A.transpose());
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_CREATE_RANDOM_MAT) {
  ASSERT_NO_THROW(sMatrix(1, 1, 475));
}

TEST(SPARSE_MATRIX, THROW_WHEN_SIZE_IS_NEGATIVE) {
  ASSERT_ANY_THROW(sMatrix(-1, 1, 475));
}

TEST(SPARSE_MATRIX, MAT_EQUAL_WITH_SAME_SEED) {
  sMatrix A = sMatrix(17, 8, 35);
  sMatrix B = sMatrix(17, 8, 35);
  ASSERT_TRUE(A == B);
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_MULTIPLY) {
  sMatrix A = sMatrix(1000, 50, 475);
  sMatrix B = sMatrix(1000, 50, 475);
  ASSERT_NO_THROW(A * B);
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_MULTIPLY_2) {
  sMatrix A = sMatrix(2000, 500, 475);
  sMatrix B = sMatrix(2000, 500, 475);
  ASSERT_NO_THROW(A * B);
}
