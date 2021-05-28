// Copyright 2021 Zaitsev Andrey
#ifndef MODULES_TASK_3_ZAITSEV_A_RADIX_SORT_RADIXSORTTBB_H_
#define MODULES_TASK_3_ZAITSEV_A_RADIX_SORT_RADIXSORTTBB_H_

#include <vector>

void merge(double *vec, double *out, int center, int size, int odd_even);
void oddEvenMergeSortSeq(double* vec, int size, int border);
void oddEvenMergeSortTbb(double* vec, int size, int border, int threads);
void radixSort(double *vec, double *out, int size);
std::vector<double> createVector(int size, double a, double b);
bool check(double* vec, int size);

#endif  // MODULES_TASK_3_ZAITSEV_A_RADIX_SORT_RADIXSORTTBB_H_
