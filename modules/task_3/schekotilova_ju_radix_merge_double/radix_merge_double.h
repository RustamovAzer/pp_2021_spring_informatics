// Copyright 2021 Schekotilova Julia
#ifndef MODULES_TASK_3_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_
#define MODULES_TASK_3_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_

#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <tbb/tbb.h>
#include <cstring>
#include <iterator>
#include <ctime>
#include <utility>
#include <random>
#include <vector>

std::vector<double> sorting(std::vector<double> in, std::vector<double> out, const int val);
std::vector<double> last_sorting(std::vector<double> in);
std::vector<double> radix_sort(const std::vector<double>& in);
std::vector<double> radix_sort_omp(const std::vector<double>& in);
std::vector<double> radix_sort_tbb(const std::vector<double>& in);
std::vector<double> merge(const std::vector<double>& arr, const std::vector<double>& arr_);
bool checker(std::vector<double> arr, int size);
std::vector<double> generate(int size);

#endif  // MODULES_TASK_3_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_
