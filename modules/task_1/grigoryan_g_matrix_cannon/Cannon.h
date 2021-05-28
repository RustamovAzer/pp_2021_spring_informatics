// Copyright 2021 Grigoryan Garry
#ifndef MODULES_TASK_1_GRIGORYAN_G_MATRIX_CANNON_CANNON_H_
#define MODULES_TASK_1_GRIGORYAN_G_MATRIX_CANNON_CANNON_H_

#include <vector>
#include <complex>
#include <iostream>
#include <cmath>

using matrix = std::vector<std::vector<double>>;

matrix RandomMatrix(const int n);

matrix NaiveMulti(const matrix &A, const matrix &B);
matrix BlockMulti(const matrix &A, const matrix &B, const int &blockSize);

bool CompareMatrix(const matrix &A, const matrix &B);
bool CompareValues(const double &a, const double &b);

#endif  // MODULES_TASK_1_GRIGORYAN_G_MATRIX_CANNON_CANNON_H_
