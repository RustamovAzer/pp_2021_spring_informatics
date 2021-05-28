// Copyright 2020 Bakaeva Maria
#ifndef MODULES_TASK_1_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
#define MODULES_TASK_1_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
#include <vector>
#include <utility>

size_t getLeftmostPoint(std::vector<std::pair<double, double>> setOfPoints);
double rotate(std::pair<double, double> A,
std::pair<double, double> B, std::pair<double, double> C);
std::vector<std::pair<double, double>>
sortPoints(std::vector<std::pair<double, double>> setOfPoints);
std::vector<std::pair<double, double>>
grahamScan(std::vector<std::pair<double, double>> setOfPoints);
#endif  // MODULES_TASK_1_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
