// Copyright 2021 Taktaev Artem
#ifndef MODULES_TASK_4_TAKTAEV_A_QSORT_QSORT_H_
#define MODULES_TASK_4_TAKTAEV_A_QSORT_QSORT_H_

#include <vector>

#include "../../../3rdparty/unapproved/unapproved.h"

std::vector<double> createRandomVector(int vec_size);
void qSortSeq(std::vector<double> *arr, int left, int right);
void merge(double* a, int size_a, double* b, int size_b);
void qSortStd(std::vector<double> *arr, int thread_count = std::thread::hardware_concurrency());

#endif  // MODULES_TASK_4_TAKTAEV_A_QSORT_QSORT_H_
