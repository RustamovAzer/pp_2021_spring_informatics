// Copyright 2021 Pronkin Dmitry
#ifndef MODULES_TASK_2_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
#define MODULES_TASK_2_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_

#define _USE_MATH_DEFINES

#include <omp.h>
#include <limits>
#include <vector>
#include <stdexcept>
#include <random>
#include <ctime>
#include <cmath>
#include <iostream>

const int KERNEL_SIZE = 3;

std::vector<double> getRandomImage(int height, int width);
std::vector<double> getGaussianKernel(double sigma);
std::vector<double> seqGaussianFilter(const std::vector<double>& image, int height, int width, double sigma);
std::vector<double> parGaussianFilter(const std::vector<double>& image, int height, int width, double sigma);

#endif  // MODULES_TASK_2_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
