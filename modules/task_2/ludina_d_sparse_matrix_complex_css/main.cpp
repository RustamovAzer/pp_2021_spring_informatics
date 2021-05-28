// Copyright 2021 Ludina Daria
#include <gtest/gtest.h>
#include "../../../modules/task_2/ludina_d_sparse_matrix_complex_css/sparse_matrix_complex_css.h"

TEST(Sparse_Matrix_CCS_1, CreateCCS) {
  std::vector<std::complex<int>> a = {
    {0, 0},  {3, 0}, {0, 0}, {7, 0},
    {0, 0}, {0, 0},  {5, 0}, {0, 0},
    {8, 0},  {0, 0},  {0, 0}, {0, 0},
    {0, 0},  {0, 0},  {1, 0}, {11, 0}
  };
  SparseMatrix SMatrix(4, 4, a);
  std::vector<int> row1 = { 2, 0, 1, 3, 0, 3 };
  std::vector<int> colidx = { 0, 1, 2, 4, 6 };
  std::vector<std::complex<int>> val = { {8, 0} , {3, 0}, {5, 0}, {1, 0}, {7, 0}, {11, 0} };
  ASSERT_EQ(val, SMatrix.value);
  ASSERT_EQ(row1, SMatrix.row);
  ASSERT_EQ(colidx, SMatrix.col_idx);
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
  ASSERT_EQ(val, SMatrixT.value);
  ASSERT_EQ(row1, SMatrixT.row);
  ASSERT_EQ(colidx, SMatrixT.col_idx);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_1) {
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

TEST(Sparse_Matrix_CCS_1, Multiplication_2) {
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

TEST(Sparse_Matrix_CCS_1, Multiplication_Parallel) {
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
  SparseMatrix result = MultiplicationParallel(A, B);
  std::vector<std::complex<int>> res = { {5, 0}, {10, -20}, {24, 32}, {8, 0}, {0, -5} };
  ASSERT_EQ(res, result.value);
}

TEST(Sparse_Matrix_CCS_1, Multiplication_Parallel_2) {
  SparseMatrix A = getRandomSparseMatrix(100, 100, 20);
  SparseMatrix B = getRandomSparseMatrix(100, 100, 40);

  SparseMatrix result_1 = Multiplication(A, B);
  SparseMatrix result_2 = MultiplicationParallel(A, B);
  ASSERT_EQ(result_1.value, result_2.value);
  ASSERT_EQ(result_1.col_idx, result_2.col_idx);
  ASSERT_EQ(result_1.row, result_2.row);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
