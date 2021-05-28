// Copyright 2021 Kirillov Konstantin
#ifndef MODULES_TASK_3_KIRILLOV_K_GAUSSIAN_FILTER_GOR_GAUS_FILTER_GOR_H_
#define MODULES_TASK_3_KIRILLOV_K_GAUSSIAN_FILTER_GOR_GAUS_FILTER_GOR_H_
#include <tbb/parallel_for.h>
#include<vector>
#include<iostream>
#include<ctime>
#include<random>
#include<algorithm>
using Matrix = std::vector<std::vector<double>>;
Matrix getRandomMatrix(int rows, int cols);
template <class T> T clamp(T tmp, int max, int min);
Matrix genMatrixKernel(int radius, double alpha);
Matrix SequentialGausFilter(Matrix img, int rows, int cols,
    int radius, double alpha);
Matrix ParallelGausFilter(Matrix img, int rows, int cols,
    int radius, double alpha);
void printMatrix(Matrix img, int rows, int cols);

#endif  // MODULES_TASK_3_KIRILLOV_K_GAUSSIAN_FILTER_GOR_GAUS_FILTER_GOR_H_
