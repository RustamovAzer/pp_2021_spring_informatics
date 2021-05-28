// Copyright 2021 Zoreev Mikhail

#ifndef MODULES_TASK_4_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#define MODULES_TASK_4_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_

#include <omp.h>

#include <functional>
#include <stdexcept>
#include <vector>

#include "../../../3rdparty/unapproved/unapproved.h"

double integrate(const std::function<double(std::vector<double>)>& f,
    const std::vector<double>& from, const std::vector<double>& to,
    uint64_t steps);

double integrate_parallel(const std::function<double(std::vector<double>)> &f,
                 const std::vector<double> &from, const std::vector<double> &to,
                 uint64_t steps, size_t threads_count);

#endif  // MODULES_TASK_4_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
