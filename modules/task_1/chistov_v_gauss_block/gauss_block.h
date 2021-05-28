// Copyright 2020 Chistov Vladimir
#ifndef MODULES_TASK_1_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_
#define MODULES_TASK_1_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_

#include <vector>

std::vector<double> Gauss_Sequential(std::vector<double> image,
    int width, int height);
std::vector <double> Block_Construct(std::vector<double> image,
    int num, int widthloc, int heightloc, int heigth);
std::vector <double> Block_Destruct(std::vector<double> empty,
    std::vector<double> block, int num, int widthloc,
int heightloc, int heigth);
std::vector<double> Gauss_Parallel(const std::vector<double> &image,
    int width, int height, int num);
std::vector<double> GenRandVec(int size);
const double Gauss_Core[9] = { 1, 2, 1,
                               2, 4, 2,
                               1, 2, 1 };

#endif  // MODULES_TASK_1_CHISTOV_V_GAUSS_BLOCK_GAUSS_BLOCK_H_
