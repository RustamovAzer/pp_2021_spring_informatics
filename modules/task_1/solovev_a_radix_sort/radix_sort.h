// Copyright 2021 Solovev Aleksandr
#ifndef MODULES_TASK_1_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#define MODULES_TASK_1_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>

int generateRandomArray(std::vector<int> *buffer, int min, int max);
int MergeArrays(std::vector<int> *buffer1, std::vector<int> *buffer2,  std::vector<int> *result);
int SortingCheck(std::vector<int> *buffer);
int RadixSortUnsigned(std::vector<int> *buffer);
int RadixSort(std::vector<int> *buffer);

#endif  // MODULES_TASK_1_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
