// Copyright 2021 Molotkova Svetlana
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include "./vert_gauss.h"

TEST(Matrix_testing, invalid_argument1) {
  int rows = 0;
  int cols = 0;
  ASSERT_ANY_THROW(matrixCreation(rows, cols));
}

TEST(Matrix_testing, invalid_argument2) {
  int rows = -1;
  int cols = 2;
  ASSERT_ANY_THROW(matrixCreation(rows, cols));
}

TEST(Gauss_filter, 1300x1200_seq_1thr) {
  // seq vs 1 thr
  int rows = 1300;
  int cols = 1200;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  omp_set_num_threads(1);
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}

TEST(Gauss_filter, 111x631_60thr) {
  // time of 60 thr
  int rows = 911;
  int cols = 691;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  omp_set_num_threads(60);
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel(60): " << parallel << std::endl;
}

TEST(Gauss_filter, 111x631_13thr) {
  // time of 13 thr
  int rows = 911;
  int cols = 691;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  omp_set_num_threads(13);
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel(13): " << parallel << std::endl;
}

TEST(Gauss_filter, 11x63_4thr) {
  // time of 4 thr
  int rows = 11;
  int cols = 63;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  omp_set_num_threads(4);
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}

TEST(Gauss_filter, 11x63_seq_thr_small) {
  // seq vs thrds
  int rows = 140;
  int cols = 140;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}

TEST(Gauss_filter, 11x63_seq_thr_big) {
  // seq vs thrds
  int rows = 1100;
  int cols = 6300;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}

TEST(Gauss_filter, 11x63_seq_thr_big_more_thr) {
  // seq vs thrds
  int rows = 1100;
  int cols = 6300;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  omp_set_num_threads(15);
  start = omp_get_wtime();
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}


TEST(Gauss_filter, pre_calc) {
  // seq vs 73 thr
  int rows = 3;
  int cols = 3;
  std::vector<double> matrix(rows * cols);
  matrix[0] = 15;
  matrix[1] = 7;
  matrix[2] = 211;
  matrix[3]= 230;
  matrix[4] = 228;
  matrix[5] = 161;
  matrix[6] = 231;
  matrix[7]= 144;
  matrix[8] = 188;
  matrix = transpose(matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> matrix_calc(rows * cols);
  matrix_calc[0] = 84;
  matrix_calc[1] = 147;
  matrix_calc[2] = 211;
  matrix_calc[3]= 120;
  matrix_calc[4] = 157;
  matrix_calc[5] = 193;
  matrix_calc[6] = 156;
  matrix_calc[7]= 166;
  matrix_calc[8] = 176;
  omp_set_num_threads(71);
  double sequential, parallel;
  double start, end;
  start = omp_get_wtime();
  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  sequential = end - start;
  start = omp_get_wtime();
  auto matrix_thr = gaussFilter_omp(matrix, rows, cols, 1, 5);
  end = omp_get_wtime();
  parallel = end - start;
  ASSERT_EQ(matrix_calc, matrix_thr);
  ASSERT_EQ(matrix_calc, matrix_seq);
  ASSERT_EQ(matrix_seq, matrix_thr);
  std::cout << "Sequential: " << sequential << "  Parallel: " << parallel << std::endl;
}

