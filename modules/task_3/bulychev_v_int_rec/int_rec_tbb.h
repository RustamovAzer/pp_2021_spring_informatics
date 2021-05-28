// Copyright 2021 Bulychev Vladislav
#ifndef MODULES_TASK_3_BULYCHEV_V_INT_REC_INT_REC_TBB_H_
#define MODULES_TASK_3_BULYCHEV_V_INT_REC_INT_REC_TBB_H_

#include <vector>

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double));
double Calculation_Tbb(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double));

#endif  // MODULES_TASK_3_BULYCHEV_V_INT_REC_INT_REC_TBB_H_
