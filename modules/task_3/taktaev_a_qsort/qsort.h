// Copyright 2021 Taktaev Artem
#ifndef MODULES_TASK_3_TAKTAEV_A_QSORT_QSORT_H_
#define MODULES_TASK_3_TAKTAEV_A_QSORT_QSORT_H_

#include <vector>

std::vector<double> createRandomVector(int vec_size);
void qSortSeq(std::vector<double> *arr, int left, int right);
void merge(double* a, int size_a, double* b, int size_b);
void qSortTbb(std::vector<double> *arr);

#endif  // MODULES_TASK_3_TAKTAEV_A_QSORT_QSORT_H_
