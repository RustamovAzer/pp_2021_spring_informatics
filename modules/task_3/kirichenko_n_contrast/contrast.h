// Copyright 2021 Kirichenko Nikita
#ifndef MODULES_TASK_3_KIRICHENKO_N_CONTRAST_CONTRAST_H_
#define MODULES_TASK_3_KIRICHENKO_N_CONTRAST_CONTRAST_H_

#include <tbb/tbb.h>
#include <tbb/tick_count.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <random>
#include <ctime>
#include <algorithm>


using Result = std::vector<int>;

Result RandomI(int l, int k);
Result Contrast(const Result& rm);
Result Contrasttbb(const Result& rm);

#endif  // MODULES_TASK_3_KIRICHENKO_N_CONTRAST_CONTRAST_H_
