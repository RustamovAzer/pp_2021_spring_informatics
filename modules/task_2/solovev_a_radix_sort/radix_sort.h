// Copyright 2021 Solovev Aleksandr
#ifndef MODULES_TASK_2_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#define MODULES_TASK_2_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>

int generateRandomArray(std::vector<int> *buffer, int min, int max);
int SortingCheck(std::vector<int> *buffer);
int RadixSort(std::vector<int> *buffer);
int ParallelSorting(std::vector<int> *buffer);
#endif  // MODULES_TASK_2_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
