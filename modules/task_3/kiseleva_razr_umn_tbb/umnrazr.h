// Copyright 2021 Kiseleva Anastasia
#ifndef MODULES_TASK_3_KISELEVA_RAZR_UMN_TBB_UMNRAZR_H_
#define MODULES_TASK_3_KISELEVA_RAZR_UMN_TBB_UMNRAZR_H_

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"

#define SIZE_ERROR -2

typedef std::vector<std::complex<int>> ComplexMatr;

ComplexMatr rand(int n, int m, int nz);
ComplexMatr transp(ComplexMatr a, int n, int m);
ComplexMatr value(ComplexMatr matr, int n, int m);
std::vector<int> rows(ComplexMatr matr, int n, int m);
std::vector<int> index(ComplexMatr matr, int n, int m);
ComplexMatr umn_posled(ComplexMatr valA, ComplexMatr valB,
    std::vector<int> rowsA, std::vector<int> rowsB,
    std::vector<int> indexA, std::vector<int> indexB,
    int na, int ma, int nb, int mb);
ComplexMatr umn_parallel(ComplexMatr valA, ComplexMatr valB,
    std::vector<int> rowsA, std::vector<int> rowsB, std::vector<int> indexA,
    std::vector<int> indexB, int na, int ma, int nb, int mb);
ComplexMatr umn_parallel_tbb(ComplexMatr valA, ComplexMatr valB,
    std::vector<int> rowsA, std::vector<int> rowsB, std::vector<int> indexA,
    std::vector<int> indexB, int na, int ma, int nb, int mb);

#endif  // MODULES_TASK_3_KISELEVA_RAZR_UMN_TBB_UMNRAZR_H_
