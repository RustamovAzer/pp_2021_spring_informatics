// Copyright 2021 Schekotilova Julia
#ifndef MODULES_TASK_1_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_
#define MODULES_TASK_1_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_
#include <time.h>
#include <random>

  void sorting(double* in, double* out, int size, int val);
  void last_sorting(double* in, double* out, int size, int val);
  void ordered_merge(double* source, double* source_, int size, int size_,
    double* res);
  void radixSort(double* source, int size);
  double* generateArray(double* arr, int size);
  bool checker(double* arr, int size);

#endif  // MODULES_TASK_1_SCHEKOTILOVA_JU_RADIX_MERGE_DOUBLE_RADIX_MERGE_DOUBLE_H_
