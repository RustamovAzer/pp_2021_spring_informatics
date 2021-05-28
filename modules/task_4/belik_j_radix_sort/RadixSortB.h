// Copyright 2021 Belik Julia
#ifndef MODULES_TASK_4_BELIK_J_RADIX_SORT_RADIXSORTB_H_
#define MODULES_TASK_4_BELIK_J_RADIX_SORT_RADIXSORTB_H_

#include <vector>

void RadixSort(double* vec, size_t len, double* vec2);
std::vector<double> Vector(size_t n, double a, double b);
std::vector<double> MergeBatcherPar(std::vector<double> vec, size_t size, int thr);
std::vector<double> MergeBatcherSeq(std::vector<double> vec, size_t size, int thr);
void Shuffle(double* vec, size_t len, double* vec2);
void PMerge(double* start1, double* start2, double* start3, size_t len1, size_t len2);

#endif  // MODULES_TASK_4_BELIK_J_RADIX_SORT_RADIXSORTB_H_
