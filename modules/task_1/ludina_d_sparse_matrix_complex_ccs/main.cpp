// Copyright 2021 Ludina Daria
#include <gtest/gtest.h>
#include "../../../modules/task_1/ludina_d_sparse_matrix_complex_ccs/sparse_matrix_complex_ccs.h"

TEST(Sparse_Matrix_CCS_1, CreateCCS) {
  std::vector<std::complex<int>> a = {
    {0, 0},  {3, 0}, {0, 0}, {7, 0},
    {0, 0}, {0, 0},  {5, 0}, {0, 0},
    {8, 0},  {0, 0},  {0, 0}, {0, 0},
    {0, 0},  {0, 0},  {14, 0}, {16, 0}
  };
  SparseMatrix SMatrix(4, 4, a);
  std::vector<int> row1 = { 2, 0, 1, 3, 0, 3 };
  std::vector<int> colidx = { 0, 1, 2, 4, 6 };
  std::vector<std::complex<int>> val = { {8, 0} , {3, 0}, {5, 0}, {14, 0}, {7, 0}, {16, 0} };
  ASSERT_EQ(SMatrix.value, val);
  ASSERT_EQ(SMatrix.row, row1);
  ASSERT_EQ(SMatrix.col_idx, colidx);
}

TEST(Sparse_Matrix_CCS_1, Transpose) {
  std::vector<std::complex<int>> a = {
    {0, 0},  {3, 0}, {0, 0}, {7, 0},
    {0, 0}, {0, 0},  {8, 0}, {0, 0},
    {0, 0},  {0, 0},  {0, 0}, {0, 0},
    {9, 0},  {0, 0},  {15, 0}, {16, 0}
  };
  SparseMatrix SMatrix(4, 4, a);
  SparseMatrix SMatrixT = SMatrix.transpose();
  std::vector<int> row1 = { 1, 3, 2, 0, 2, 3 };
  std::vector<int> colidx = { 0, 2, 3, 3, 6 };
  std::vector<std::complex<int>> val = { {3, 0} , {7, 0}, {8, 0}, {9, 0}, {15, 0}, {16, 0} };
  ASSERT_EQ(SMatrixT.value, val);
  ASSERT_EQ(SMatrixT.row, row1);
  ASSERT_EQ(SMatrixT.col_idx, colidx);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_0) {
  std::vector<std::complex<int>> a = {
    {1, 0}, {0, 0},
    {0, 0}, {2, 0} };
  SparseMatrix A(2, 2, a);
  std::vector<std::complex<int>> b = {
    {0, 0}, {0, 0},
    {0, 0}, {0, 0} };
  SparseMatrix B(2, 2, b);
  SparseMatrix result = Multiplication(A, B);
  std::vector<std::complex<int>> res = {};
  ASSERT_EQ(res, result.value);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_1) {
  std::vector<std::complex<int>> a = {
    {1, 0}, {0, 0},
    {0, 0}, {2, 0}};
  SparseMatrix A(2, 2, a);
  std::vector<std::complex<int>> b = {
    {7, 0}, {5, 0},
    {0, 0}, {0, 0}};
  SparseMatrix B(2, 2, b);
  SparseMatrix result = Multiplication(A, B);
  std::vector<std::complex<int>> res = { {7, 0}, {5, 0} };
  ASSERT_EQ(res, result.value);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_2) {
  std::vector<std::complex<int>> a = {
    {0, 0}, {12, 0}, {0, 0},
    {1, 0}, {0, 0}, {5, 0},
    {0, 0}, {4, 0}, {7, 0} };
  SparseMatrix A(3, 3, a);
  std::vector<std::complex<int>> b = {
    {3, 0}, {0, 0}, {9, 0},
    {0, 0}, {2, 0}, {0, 0},
    {0, 0}, {0, 0}, {1, 0} };
  SparseMatrix B(3, 3, b);
  SparseMatrix result = Multiplication(A, B);
  std::vector<std::complex<int>> res = { {3, 0}, {24, 0}, {8, 0}, {14, 0}, {7, 0} };
  ASSERT_EQ(res, result.value);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_3) {
  std::vector<std::complex<int>> a = {
    {0, 0}, {5, 0}, {0, 0},
    {3, 4}, {0, 0}, {1, -2},
    {1, 0}, {0, 0}, {0, 0} };
  SparseMatrix A(3, 3, a);
  std::vector<std::complex<int>> b = {
    {0, 0}, {8, 0}, {0, 0},
    {1, 0}, {2, -4}, {0, 0},
    {0, 0}, {0, 0}, {2, -1} };
  SparseMatrix B(3, 3, b);
  SparseMatrix result = Multiplication(A, B);
  std::vector<std::complex<int>> res = { {5, 0}, {10, -20}, {24, 32}, {8, 0}, {0, -5} };
  ASSERT_EQ(res, result.value);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
