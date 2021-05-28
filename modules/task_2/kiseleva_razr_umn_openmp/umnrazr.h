// Copyright 2021 Kiseleva Anastasia
#ifndef MODULES_TASK_2_KISELEVA_RAZR_UMN_OPENMP_UMNRAZR_H_
#define MODULES_TASK_2_KISELEVA_RAZR_UMN_OPENMP_UMNRAZR_H_

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <ctime>

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

#endif  // MODULES_TASK_2_KISELEVA_RAZR_UMN_OPENMP_UMNRAZR_H_
