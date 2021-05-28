// Copyright 2021 Pronkin Dmitry
#ifndef MODULES_TASK_4_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
#define MODULES_TASK_4_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_

#include <vector>

std::vector<double> getRandomImage(int height, int width);
std::vector<double> getGaussianKernel(double sigma);
std::vector<double> seqGaussianFilter(const std::vector<double>& image, int height, int width, double sigma);
std::vector<double> parGaussianFilter(const std::vector<double>& image, int height, int width, double sigma);

#endif  // MODULES_TASK_4_PRONKIN_D_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
