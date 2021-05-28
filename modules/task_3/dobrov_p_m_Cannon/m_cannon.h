// Copyright 2021 Dobrov Pavel
#ifndef MODULES_TASK_3_DOBROV_P_M_CANNON_M_CANNON_H_
#define MODULES_TASK_3_DOBROV_P_M_CANNON_M_CANNON_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

std::vector<double> RandomMatrix(int n);
std::vector<double> parMulti(const std::vector<double> &A, const std::vector<double> &B, int n);
std::vector<double> seqMulti(const std::vector<double> &A, const std::vector<double> &B, int n);

#endif  // MODULES_TASK_3_DOBROV_P_M_CANNON_M_CANNON_H_
