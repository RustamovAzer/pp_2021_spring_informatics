// Copyright 2021 Kolesin Andrey
#ifndef MODULES_TASK_4_KOLESIN_A_RADIX_SORT_RADIX_H_
#define MODULES_TASK_4_KOLESIN_A_RADIX_SORT_RADIX_H_

#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include <iostream>

#include "../../../3rdparty/unapproved/unapproved.h"
// #include <chrono>


void radixSort(int* arr, int size);
std::vector<int> getRandomVector(int size = -1, int maxEl = -1);
void printVec(std::vector<int> vec);
void parallelSort(int* arr, int size, int threads);

#endif  // MODULES_TASK_4_KOLESIN_A_RADIX_SORT_RADIX_H_
