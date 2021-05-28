// Copyright 2021 Makarychev Sergey
#ifndef  MODULES_TASK_3_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
#define MODULES_TASK_3_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
#include <iostream>
#include <vector>

std::vector<int> getRandomVector(int sizeVec);
void createCounters(int* sortVec, int* counters, int sizeVec);
void signedRadixSort(std::vector<int>* sortVec);
void countersSort(int numByte, std::vector<int>* sortVec,
  std::vector<int>* output);
void unsignedRadixSort(std::vector<int>* sortVec);
void signedRadixSortParallel(std::vector<int>* sortVec,
  int leftIndex, int rightIndex);
void mergeOrderVec(int* vec1, int size1, int* vec2, int size2);
void signedRadixSortTbb(std::vector<int>* sortVec);

#endif  // MODULES_TASK_3_MAKARYCHEV_S_BITWISE_SORT_BITWISE_SORT_H_
