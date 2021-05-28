// Copyright 2021 Gushchin Artem
#ifndef MODULES_TASK_1_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_
#define MODULES_TASK_1_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_

#include <utility>
#include <vector>

double simpsonsMethod(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<std::pair<double, double>>& segments,
    int stepsCount);

#endif  // MODULES_TASK_1_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_
