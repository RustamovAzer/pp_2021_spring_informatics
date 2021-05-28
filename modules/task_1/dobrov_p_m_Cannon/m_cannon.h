// Copyright 2021 Dobrov Pavel
#ifndef MODULES_TASK_1_DOBROV_P_M_CANNON_M_CANNON_H_
#define MODULES_TASK_1_DOBROV_P_M_CANNON_M_CANNON_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#define THREADS 4;

std::vector<double> RandomMatrix(int n, int m);
std::vector<double> MatrixMulti(const std::vector<double> &A, const std::vector<double> &B, int m, int n, int l);
std::vector<double> NaiveMulti(const std::vector<double> &A, const std::vector<double> &B, int m, int n, int l);

#endif  // MODULES_TASK_1_DOBROV_P_M_CANNON_M_CANNON_H_
