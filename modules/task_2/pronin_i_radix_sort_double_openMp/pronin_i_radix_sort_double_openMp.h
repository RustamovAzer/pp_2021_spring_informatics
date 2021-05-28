// Copyright 2021 Pronin Igor
#ifndef MODULES_TASK_2_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_H_
#define MODULES_TASK_2_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_H_
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
void RandVec(double*, int, int, int);
void SortBite(double*, double*, int, int);
void SortAll(double*, double*, int);
void ParSort(double*, double*, int, int);
void SeqSort(double*, double*, int, int);
void Merge(double* mas1, double* mas2, double* mas3, int size1, int size2);
#endif  // MODULES_TASK_2_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_PRONIN_I_RADIX_SORT_DOUBLE_OPENMP_H_
