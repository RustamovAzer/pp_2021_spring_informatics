// Copyright 2021 Kasjanychev Mikhail
#ifndef MODULES_TASK_1_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_
#define MODULES_TASK_1_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_

#include <iostream>
#include <vector>
#include <cassert>

std::vector<double> multiMtxVec(std::vector<std::vector<double> >, std::vector<double>);
std::vector<double> subVec(std::vector<double>, std::vector<double>);
std::vector<double> sumVec(std::vector<double>, std::vector<double>);
double scalarProduct(std::vector<double>, std::vector<double>);
std::vector<double> multiVec(double, std::vector<double>);
std::vector<double> minusVec(std::vector<double>);
std::vector<double> calculateRes(std::vector<std::vector<double> >, const std::vector<double>&);

#endif  // MODULES_TASK_1_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_
