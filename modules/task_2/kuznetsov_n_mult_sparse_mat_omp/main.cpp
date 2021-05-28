// Copyright 2021 Kuznetsov Nikita
#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include "../../../modules/task_2/kuznetsov_n_mult_sparse_mat_omp/sparse_matrix.h"

TEST(SPARSE_MATRIX, MAT_EQUAL_ITSELF_1) {
  std::vector<double> arg1 = { 1, 1, 1 };
  std::vector<int> arg2 = { 0, 1, 2, 3 };
  std::vector<int> arg3 = { 0, 1, 2 };
  sMatrix A(arg1, arg2, arg3, 3, 3);
  ASSERT_TRUE(A == A);
}

TEST(SPARSE_MATRIX, MAT_EQUAL_ITSELF_2) {
  std::vector<double> arg1 = { 5, 8, 1, 2, 6 };
  std::vector<int> arg2 = { 0, 2, 3, 5, 5 };
  std::vector<int> arg3 = { 0, 3, 2, 1, 3 };
  sMatrix A(arg1, arg2, arg3, 5, 4);
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

TEST(SPARSE_MATRIX, CORRECT_MULT_IDENTITY_MATRIX) {
  sMatrix A({ 1, 1, 1 }, { 0, 1, 2, 3 }, { 0, 1, 2 }, 3, 3);
  sMatrix B({ 3, 5, 1 }, { 0, 1, 2, 3 }, { 0, 1, 2 }, 3, 3);
  ASSERT_TRUE(B == A * B);
}

TEST(SPARSE_MATRIX, CORRECT_MULT) {
  sMatrix A({ 1, 1 }, { 0, 1, 1, 2 }, { 0, 2 }, 2, 3);
  sMatrix B({ 7, 8, 9 }, { 0, 1, 2, 3 }, { 0, 1, 2 }, 3, 3);
  sMatrix exp_result({ 7, 9 }, { 0, 1, 1, 2 }, { 0, 2 }, 2, 3);
  sMatrix result = A * B;
  ASSERT_TRUE(exp_result == result);
}

TEST(SPARSE_MATRIX, MAT_EQUAL_WITH_SAME_SEED) {
  sMatrix A(17, 8, 35);
  sMatrix B(17, 8, 35);
  ASSERT_TRUE(A == B);
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_MULTIPLY) {
  sMatrix A(1000, 50, 10);
  sMatrix B(1000, 50, 10);
  ASSERT_NO_THROW(A * B);
}

TEST(SPARSE_MATRIX, NO_THROW_WHEN_MULTIPLY_2) {
  sMatrix A = sMatrix(2000, 500, 475);
  sMatrix B = sMatrix(2000, 500, 475);
  ASSERT_NO_THROW(A * B);
}

TEST(SPARSE_MATRIX, PARALLEL_MULT_MATRIX_COMPARE_SEQ_1000_ELEM) {
  sMatrix A(1000, 100, 10);
  sMatrix B(1000, 100, 10);

  double time1 = omp_get_wtime();
  sMatrix result = A * B;
  double time2 = omp_get_wtime();
  std::cout << "SEQ TIME: " << time2 - time1 << std::endl;

  time1 = omp_get_wtime();
  sMatrix par_result = A.pMult(B, 4);
  time2 = omp_get_wtime();
  std::cout << "PAR TIME: " << time2 - time1 << std::endl;

  bool f = true;
  for (size_t i = 0; i < result.val.size(); i++) {
    if (fabs(result.val[i] - par_result.val[i]) >=
      std::numeric_limits<double>::epsilon()) {
      f = false;
      break;
    }
  }

  ASSERT_TRUE(f);
}

TEST(SPARSE_MATRIX, PARALLEL_MULT_MATRIX_COMPARE_SEQ_2000_ELEM) {
  sMatrix A(2000, 200, 10);
  sMatrix B(2000, 200, 10);

  double time1 = omp_get_wtime();
  sMatrix result = A * B;
  double time2 = omp_get_wtime();
  std::cout << "SEQ TIME: " << time2 - time1 << std::endl;

  time1 = omp_get_wtime();
  sMatrix par_result = A.pMult(B, 4);
  time2 = omp_get_wtime();
  std::cout << "PAR TIME: " << time2 - time1 << std::endl;

  bool f = true;
  for (size_t i = 0; i < result.val.size(); i++) {
    if (fabs(result.val[i] - par_result.val[i]) >=
      std::numeric_limits<double>::epsilon()) {
      f = false;
      break;
    }
  }

  ASSERT_TRUE(f);
}

TEST(SPARSE_MATRIX, PARALLEL_MULT_MATRIX_COMPARE_SEQ_4000_ELEM) {
  sMatrix A(4000, 300, 10);
  sMatrix B(4000, 300, 10);

  double time1 = omp_get_wtime();
  sMatrix result = A * B;
  double time2 = omp_get_wtime();
  std::cout << "SEQ TIME: " << time2 - time1 << std::endl;

  time1 = omp_get_wtime();
  sMatrix par_result = A.pMult(B, 4);
  time2 = omp_get_wtime();
  std::cout << "PAR TIME: " << time2 - time1 << std::endl;

  bool f = true;
  for (size_t i = 0; i < result.val.size(); i++) {
    if (fabs(result.val[i] - par_result.val[i]) >=
      std::numeric_limits<double>::epsilon()) {
      f = false;
      break;
    }
  }

  ASSERT_TRUE(f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
