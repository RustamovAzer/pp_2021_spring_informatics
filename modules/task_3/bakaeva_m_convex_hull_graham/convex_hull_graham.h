// Copyright 2021 Bakaeva Maria
#ifndef MODULES_TASK_3_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
#define MODULES_TASK_3_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
#include <vector>
#include <utility>

std::vector<std::pair<double, double>> generateRandomPoints(size_t count);

size_t getLeftmostPoint(std::vector<std::pair<double, double>> setOfPoints);

double rotate(std::pair<double, double> A,
    std::pair<double, double> B, std::pair<double, double> C);

std::vector<std::pair<double, double>>
sortPoints(std::vector<std::pair<double, double>> setOfPoints);

std::vector<std::pair<double, double>> grahamScan(
std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e);

std::vector<std::pair<double, double> > grahamScanParallel(
    std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e,
    int numberOfThreads);
#endif  // MODULES_TASK_3_BAKAEVA_M_CONVEX_HULL_GRAHAM_CONVEX_HULL_GRAHAM_H_
