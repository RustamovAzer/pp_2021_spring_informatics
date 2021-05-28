// Copyright 2021 Makarychev Sergey
#ifndef  MODULES_TASK_1_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
#define MODULES_TASK_1_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
#include <iostream>
#include <vector>

std::vector<int> getRandomVector(int sizeVec);
void createCounters(int* sortVec, int* counters, int sizeVec);
void signedRadix(int byteNumber, int sizeVec, int* sourceVec, int* destVec, int* count);
void signedRadixSort(int* sortVec, int sizeVec);

#endif  // MODULES_TASK_1_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
