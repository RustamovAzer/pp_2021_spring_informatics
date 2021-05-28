// Copyright 2021 Denis Ostapovich
#ifndef MODULES_TASK_3_OSTAPOVICH_D_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H_
#define MODULES_TASK_3_OSTAPOVICH_D_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H_

#include<vector>
#include<cstdint>

std::vector<double> calculateCore(int size, double deviation = 1.0);
std::vector<uint8_t> getRandomMatrix(int width, int height);
std::vector<uint8_t> filter(const std::vector<uint8_t>& matrix, int width, int coreSize = 3);
std::vector<uint8_t> filterParallel(const std::vector<uint8_t>& matrix, int width, int coreSize = 3);

#endif  // MODULES_TASK_3_OSTAPOVICH_D_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H_
