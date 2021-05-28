// Copyright 2021 Zaitsev Andrey
#ifndef MODULES_TASK_2_ZAITSEV_A_RADIX_SORT_RADIXSORTOMP_H_
#define MODULES_TASK_2_ZAITSEV_A_RADIX_SORT_RADIXSORTOMP_H_

#include <omp.h>
#include <vector>

void merge(double *vec, double *out, int center, int size, int odd_even);
void oddEvenMergeSortSeq(double* vec, int size, int border);
void oddEvenMergeSortOmp(double* vec, int size, int border, int threads);
void radixSort(double *vec, double *out, int size);
std::vector<double> createVector(int size, double a, double b);
bool check(double* vec, int size);

#endif  // MODULES_TASK_2_ZAITSEV_A_RADIX_SORT_RADIXSORTOMP_H_
