// Copyright 2021 Bulychev Vladislav
#ifndef MODULES_TASK_2_BULYCHEV_V_INT_REC_INT_REC_OMP_H_
#define MODULES_TASK_2_BULYCHEV_V_INT_REC_INT_REC_OMP_H_

#include <omp.h>
#include <vector>

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>));
double Calculation_Omp(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>));

#endif  // MODULES_TASK_2_BULYCHEV_V_INT_REC_INT_REC_OMP_H_
