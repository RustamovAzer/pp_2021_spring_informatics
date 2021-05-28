// Copyright 2021 Gruzdeva Diana

#include "../../../modules/task_2/gruzdeva_d_fox_mult/fox_mult.h"
#include <omp.h>
#include <random>
#include <vector>

std::vector<double> getRandomMatrix(int size, time_t seed) {
  std::vector<double> matrix(size * size);
  std::mt19937 gen((unsigned int)seed);
  std::uniform_real_distribution<double> dis(-10, 10);

  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }

  return matrix;
}

std::vector<double> directMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB) {
  if (matrixA.size() != matrixB.size()) {
    throw "Matrices sizes differ";
  }

  int size = static_cast<int>(sqrt(matrixA.size()));
  std::vector<double> matrixC(size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrixC[i * size + j] = 0;
      for (int k = 0; k < size; k++) {
        matrixC[i * size + j] += matrixA[i * size + k]
        * matrixB[k * size + j];
      }
    }
  }

  return matrixC;
}

std::vector<double> sequentialFoxMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int blockSize) {
  if (blockSize == 0) {
    throw "Block size cannot be zero";
  }

  if (blockSize < 0) {
    throw "Block size cannot be negative";
  }

  if (matrixA.size() < (unsigned int)(blockSize * blockSize)) {
    throw "Block size cannot be larger than size of original matrices";
  }

  if (matrixA.size() != matrixB.size()) {
    throw "Matrice's sizes differ";
  }

  if (static_cast<int>(sqrt(matrixA.size())) % blockSize != 0) {
    throw "Cannot multiply matrices using this block size";
  }

  int size = static_cast<int>(sqrt(matrixA.size()));
  std::vector<double> matrixC(size * size);
  int blockCount = size / blockSize;

  for (int i = 0; i < size * size; i++) {
    matrixC[i] = 0;
  }

  for (int i = 0; i < size; i += blockCount) {
    for (int j = 0; j < size; j += blockCount) {
      for (int k = 0; k < size; k += blockCount) {
        for (int ii = i; ii < std::min(size, i + blockCount); ii++) {
          for (int jj = j; jj < std::min(size, j + blockCount); jj++) {
            for (int kk = k; kk < std::min(size, k + blockCount); kk++) {
              matrixC[ii * size + jj] += matrixA[ii * size + kk]
              * matrixB[kk * size + jj];
            }
          }
        }
      }
    }
  }
  return matrixC;
}

std::vector<double> parallelFoxMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int threadCount) {
  if (threadCount == 0) {
    throw "Number of threads cannot be zero";
  }

  if (threadCount < 0) {
    throw "Number of threads cannot be negative";
  }

  if (matrixA.size() != matrixB.size()) {
    throw "Matrice's sizes differ";
  }

  if (matrixA.size() %  static_cast<int>(sqrt(threadCount)) != 0) {
    throw "Cannot distribute data between threads";
  }

  int size = static_cast<int>(sqrt(matrixA.size()));
  int gridSize = static_cast<int>(sqrt(threadCount));
  int blockSize = size / gridSize;
  std::vector<double> matrixC(size * size);

  #pragma omp parallel
  {
    int threadNumber = omp_get_thread_num();
    int ii = threadNumber / static_cast<int>(sqrt(threadCount));
    int jj = threadNumber %  static_cast<int>(sqrt(threadCount));

    for (int step = 0; step < gridSize; step++) {
      for (int i = ii * blockSize; i < (ii + 1) * blockSize; i++) {
        for (int j = jj * blockSize; j < (jj + 1) * blockSize; j++) {
          for (int k = step * blockSize; k < (step + 1) * blockSize; k++) {
            matrixC[i * size + j] += matrixA[i * size + k] *
            matrixB[k * size + j];
          }
        }
      }
    }
  }

  return matrixC;
}
