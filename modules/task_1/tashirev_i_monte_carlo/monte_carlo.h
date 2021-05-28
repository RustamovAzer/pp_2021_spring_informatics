// Copyright 2021 Tashirev Ivan
#ifndef MODULES_TASK_1_TASHIREV_I_MONTE_CARLO_MONTE_CARLO_H_
#define MODULES_TASK_1_TASHIREV_I_MONTE_CARLO_MONTE_CARLO_H_

#include <vector>
#include <string>

double seqMonteCarlo(double(*f)(std::vector<double>), const std::vector<double>&a,
    const std::vector<double>&b, int s);

#endif  // MODULES_TASK_1_TASHIREV_I_MONTE_CARLO_MONTE_CARLO_H_
