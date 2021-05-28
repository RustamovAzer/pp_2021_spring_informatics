// Copyright 2021 Gruzdeva Diana

#ifndef MODULES_TASK_2_GRUZDEVA_D_FOX_MULT_FOX_MULT_H_
#define MODULES_TASK_2_GRUZDEVA_D_FOX_MULT_FOX_MULT_H_

#include <omp.h>
#include <vector>
#include <ctime>

std::vector<double> getRandomMatrix(int size, time_t seed);
std::vector<double> directMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB);
std::vector<double> sequentialFoxMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int blockSize);
std::vector<double> parallelFoxMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int threadCount);
#endif  // MODULES_TASK_2_GRUZDEVA_D_FOX_MULT_FOX_MULT_H_
