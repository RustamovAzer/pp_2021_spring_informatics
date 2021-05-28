// Copyright 2021 Molotkova Svetlana
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include <tbb/blocked_range.h>
#include <limits.h>
#include <iostream>
#include <vector>

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

TEST(Gauss_filter, 1300x1200) {
  int rows = 1300;
  int cols = 1200;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);

  std::vector<double> seg_begin = {0, 0};
  std::vector<double> seg_end = {1, 1};
  std::pair<tbb::tick_count, tbb::tick_count> time;
  time.first = tbb::tick_count::now();

  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Sequential " << (time.second - time.first).seconds() << std::endl;
  time.first = tbb::tick_count::now();

  auto matrix_thr = gaussFilter_par(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Parallel " << (time.second - time.first).seconds() << std::endl;
  // 5
  ASSERT_EQ(matrix_seq, matrix_thr);
}

TEST(Gauss_filter, 2500x2500) {
  int rows = 2500;
  int cols = 2500;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> seg_begin = {0, 0};
  std::vector<double> seg_end = {1, 1};
  std::pair<tbb::tick_count, tbb::tick_count> time;
  time.first = tbb::tick_count::now();

  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Sequential " << (time.second - time.first).seconds() << std::endl;
  time.first = tbb::tick_count::now();

  auto matrix_thr = gaussFilter_par(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Parallel " << (time.second - time.first).seconds() << std::endl;
  // 2,5
  ASSERT_EQ(matrix_seq, matrix_thr);
}

TEST(Gauss_filter, 1500x1700) {
  int rows = 1500;
  int cols = 1700;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  std::swap(cols, rows);
  std::vector<double> seg_begin = {0, 0};
  std::vector<double> seg_end = {1, 1};
  std::pair<tbb::tick_count, tbb::tick_count> time;
  time.first = tbb::tick_count::now();

  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Sequential " << (time.second - time.first).seconds() << std::endl;
  time.first = tbb::tick_count::now();

  auto matrix_thr = gaussFilter_par(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Parallel " << (time.second - time.first).seconds() << std::endl;
  ASSERT_EQ(matrix_seq, matrix_thr);
}

TEST(Gauss_filter, 100x1500) {
  int rows = 100;
  int cols = 1500;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  std::swap(cols, rows);
  std::vector<double> seg_begin = {0, 0};
  std::vector<double> seg_end = {1, 1};
  std::pair<tbb::tick_count, tbb::tick_count> time;
  time.first = tbb::tick_count::now();

  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Sequential " << (time.second - time.first).seconds() << std::endl;
  time.first = tbb::tick_count::now();

  auto matrix_thr = gaussFilter_par(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Parallel " << (time.second - time.first).seconds() << std::endl;
  ASSERT_EQ(matrix_seq, matrix_thr);
}

TEST(Gauss_filter, 15000x10000) {
  int rows = 1500;
  int cols = 1000;
  auto rand_matrix = matrixCreation(rows, cols);
  auto matrix = transpose(rand_matrix, rows, cols);
  std::swap(cols, rows);
  std::swap(cols, rows);
  std::vector<double> seg_begin = {0, 0};
  std::vector<double> seg_end = {1, 1};
  std::pair<tbb::tick_count, tbb::tick_count> time;
  time.first = tbb::tick_count::now();

  auto matrix_seq = gaussFilter(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Sequential " << (time.second - time.first).seconds() << std::endl;
  time.first = tbb::tick_count::now();

  auto matrix_thr = gaussFilter_par(matrix, rows, cols, 1, 5);

  time.second = tbb::tick_count::now();
  std::cout << "Parallel " << (time.second - time.first).seconds() << std::endl;
  ASSERT_EQ(matrix_seq, matrix_thr);
}
