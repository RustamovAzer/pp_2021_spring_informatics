// Copyright 2021 Molotkova Svetlana
#include <gtest/gtest.h>
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

TEST(Gauss_filter, 1x1) {
  int rows = 1;
  int cols = 1;
  std::vector<double> matrix(rows * cols);
  matrix[0] = 221;
  matrix = transpose(matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> matrix_calc(rows * cols);
  matrix_calc[0] = 220;
  ASSERT_EQ(matrix_calc, gaussFilter(matrix, rows, cols, 1, 5));
}

TEST(Gauss_filter, 5x3) {
  int rows = 5;
  int cols = 3;
  std::vector<double> matrix(rows * cols);
  matrix[0] = 221;
  matrix[1] = 177;
  matrix[2] = 22;
  matrix[3]= 47;
  matrix[4] = 77;
  matrix[5] = 55;
  matrix[6] = 12;
  matrix[7]= 56;
  matrix[8] = 78;
  matrix[9] = 72;
  matrix[10] = 28;
  matrix[11] = 55;
  matrix[12] = 19;
  matrix[13] = 0;
  matrix[14] = 133;
  matrix = transpose(matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> matrix_calc(rows * cols);
  matrix_calc[0] = 156;
  matrix_calc[1] = 96;
  matrix_calc[2] = 47;
  matrix_calc[3]= 32;
  matrix_calc[4] = 27;
  matrix_calc[5] = 113;
  matrix_calc[6] = 82;
  matrix_calc[7]= 53;
  matrix_calc[8] = 50;
  matrix_calc[9] = 51;
  matrix_calc[10] = 69;
  matrix_calc[11] = 68;
  matrix_calc[12] = 59;
  matrix_calc[13] = 68;
  matrix_calc[14] = 74;
  ASSERT_EQ(matrix_calc, gaussFilter(matrix, rows, cols, 1, 5));
}

TEST(Gauss_filter, 8x1) {
  int rows = 8;
  int cols = 1;
  std::vector<double> matrix(rows * cols);
  matrix[0] = 21;
  matrix[1] = 67;
  matrix[2] = 2;
  matrix[3]= 2;
  matrix[4] = 77;
  matrix[5] = 55;
  matrix[6] = 12;
  matrix[7]= 56;
  matrix = transpose(matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> matrix_calc(rows * cols);
  matrix_calc[0] = 36;
  matrix_calc[1] = 29;
  matrix_calc[2] = 23;
  matrix_calc[3]= 27;
  matrix_calc[4] = 44;
  matrix_calc[5] = 48;
  matrix_calc[6] = 40;
  matrix_calc[7]= 41;
  ASSERT_EQ(matrix_calc, gaussFilter(matrix, rows, cols, 1, 5));
}

TEST(Gauss_filter, 4x3) {
  int rows = 4;
  int cols = 3;
  std::vector<double> matrix(rows * cols);
  matrix[0] = 105;
  matrix[1] = 177;
  matrix[2] = 111;
  matrix[3]= 200;
  matrix[4] = 28;
  matrix[5] = 151;
  matrix[6] = 230;
  matrix[7]= 44;
  matrix[8] = 88;
  matrix[9] = 112;
  matrix[10] = 128;
  matrix[11] = 155;
  matrix = transpose(matrix, rows, cols);
  std::swap(cols, rows);
  std::vector<double> matrix_calc(rows * cols);
  matrix_calc[0] = 133;
  matrix_calc[1] = 146;
  matrix_calc[2] = 142;
  matrix_calc[3]= 134;
  matrix_calc[4] = 129;
  matrix_calc[5] = 125;
  matrix_calc[6] = 126;
  matrix_calc[7]= 128;
  matrix_calc[8] = 125;
  matrix_calc[9] = 105;
  matrix_calc[10] = 109;
  matrix_calc[11] = 121;
  ASSERT_EQ(matrix_calc, gaussFilter(matrix, rows, cols, 1, 5));
}

TEST(Gauss_filter, 3x3) {
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
  ASSERT_EQ(matrix_calc, gaussFilter(matrix, rows, cols, 1, 5));
}
