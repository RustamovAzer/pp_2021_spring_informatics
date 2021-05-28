// Copyright 2021 Molotkova Svetlana
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "../../../modules/task_3/molotkova_s_tbb/vert_gauss.h"

int clamp(int n, int upper, int lower) {
  return n <= lower ? lower : n >= upper ? upper : n;
}

std::vector<double> matrixCreation(int rows, int cols) {
  if ((rows <= 0) || (cols <= 0))
     throw std::invalid_argument("Matrix can't be created");
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::vector<double> result(rows * cols);
  for (int i = 0; i < rows * cols; i++)
    result[i] = gen() % 256;
  return result;
}

std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
  std::vector<double> result(rows * cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i + j * rows] = matrix[i * cols + j];
    }
  }
  return result;
}

std::vector<double> gaussKernel(int radius, int sigma) {
  // determine the size of the kernel = 3б, radius = 1
  const int size = 2 * radius + 1;
  // coef of normalization of kernel
  double norm = 0;
  // creating gauss kernel
  std::vector<double> kernel(size * size);
  // calculating linear filter kernel
  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      int idx = (i + radius) * size + j + radius;
      kernel[idx] = exp(-(i * i + j * j) / (sigma * sigma));
      norm += kernel[idx];
    }
  }

  // normalizing the kernel
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
        kernel[i * size + j] /= norm;
  return kernel;
}

std::vector<double> gaussFilter(const std::vector<double>& matrix, int rows, int cols,
  int radius, int sigma) {
  std::vector<double> resultMatrix(rows * cols);
  const unsigned int size = 2 * radius + 1;
  std::vector<double> kernel = gaussKernel(radius, sigma);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      double res = 0;
      for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
          int idx = (i + radius) * size + j + radius;
          int x_ = clamp(x + j, rows - 1, 0);
          int y_ = clamp(y + i, cols - 1, 0);
          double value = matrix[x_ * cols + y_];
          res += value * kernel[idx];
        }
      }
      res = clamp(res, 255, 0);
      resultMatrix[x * cols + y] = res;
    }
  }
  return resultMatrix;
}

std::vector<double> gaussFilter_par(const std::vector<double>& matrix, int rows, int cols,
  int radius, int sigma) {
  std::vector<double> resultMatrix(rows * cols);
  const unsigned int size = 2 * radius + 1;
  std::vector<double> kernel = gaussKernel(radius, sigma);
      tbb::parallel_for(0, rows, [&](int x) {
          // tbb::parallel_for (0, cols, [&](int y) {
          for (int y = 0; y < cols; y++) {
          double res = 0;
          for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
              int idx = (i + radius) * size + j + radius;
              int x_ = clamp(x + j, rows - 1, 0);
              int y_ = clamp(y + i, cols - 1, 0);
              double value = matrix[x_ * cols + y_];
              res += value * kernel[idx];
            }
          }
          res = clamp(res, 255, 0);
          resultMatrix[x * cols + y] = res;
        // });
          }
      });
  return resultMatrix;
  }
