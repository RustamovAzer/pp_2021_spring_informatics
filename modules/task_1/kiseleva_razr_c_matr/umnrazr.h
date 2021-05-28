// Copyright 2021 Kiseleva Anastasia
#ifndef MODULES_TASK_1_KISELEVA_RAZR_C_MATR_UMNRAZR_H_
#define MODULES_TASK_1_KISELEVA_RAZR_C_MATR_UMNRAZR_H_

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
ComplexMatr umn_posled(ComplexMatr A, const ComplexMatr& B,
    int na, int ma, int nb, int mb);
ComplexMatr proverka(ComplexMatr a, ComplexMatr b,
    int na, int ma, int nb, int mb);

#endif  // MODULES_TASK_1_KISELEVA_RAZR_C_MATR_UMNRAZR_H_
