// Copyright 2021 Kirichenko Nikita
#ifndef MODULES_TASK_2_KIRICHENKO_N_CONTRAST_CONTRAST_H_
#define MODULES_TASK_2_KIRICHENKO_N_CONTRAST_CONTRAST_H_

#include <omp.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <ctime>
#include <algorithm>


using Result = std::vector<int>;

Result RandomI(int l, int k);
Result Contrast(const Result& rm);
Result Contrastomp(const Result& rm);

#endif  // MODULES_TASK_2_KIRICHENKO_N_CONTRAST_CONTRAST_H_
