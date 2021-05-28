// Copyright 2021 Kirichenko Nikita
#ifndef MODULES_TASK_1_KIRICHENKO_N_CONTRAST_CONTRAST_H_
#define MODULES_TASK_1_KIRICHENKO_N_CONTRAST_CONTRAST_H_

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <ctime>
#include <algorithm>

using Result = std::vector<int>;

Result RandomM(int l, int k);
Result Contrast(const Result& rm);


#endif  // MODULES_TASK_1_KIRICHENKO_N_CONTRAST_CONTRAST_H_
