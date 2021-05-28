// Copyright 2021 Chistov Vladimir
#ifndef MODULES_TASK_2_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_
#define MODULES_TASK_2_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_

#include <vector>

std::vector<double> Gauss_Sequential(std::vector<double> image,
    int width, int height);
std::vector<double> Gauss_Parallel(std::vector<double> image,
    int width, int height);
std::vector<double> GenRandVec(int size);
std::vector<double> Gauss_Core(int size, double dev = 1.0);

#endif  // MODULES_TASK_2_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_
