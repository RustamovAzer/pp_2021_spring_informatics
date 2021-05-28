// Copyright 2021 Zoreev Mikhail

#ifndef MODULES_TASK_1_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#define MODULES_TASK_1_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_

#include <functional>
#include <stdexcept>
#include <vector>

double integrate(const std::function<double(std::vector<double>)>& f,
    const std::vector<double>& from, const std::vector<double>& to,
    uint64_t steps);

#endif  // MODULES_TASK_1_ZOREEV_M_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
