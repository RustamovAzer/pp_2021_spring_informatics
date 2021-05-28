// Copyright 2021 Bakaeva Maria
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <stack>

#include "../../../modules/task_1/bakaeva_m_convex_hull_graham/convex_hull_graham.h"

size_t getLeftmostPoint(std::vector<std::pair<double, double>> setOfPoints) {
    size_t n = setOfPoints.size();
    size_t leftmostPoint = 0;

    for (size_t i = 0; i < n; i++) {
        if (setOfPoints[i] < setOfPoints[leftmostPoint]) {
            leftmostPoint = i;
        }
    }

    return leftmostPoint;
}

double rotate(std::pair<double, double> A,
std::pair<double, double> B, std::pair<double, double> C) {
    return (B.first - A.first) * (C.second - B.second)
    - (B.second - A.second) * (C.first - B.first);
}

std::vector<std::pair<double, double>>
sortPoints(std::vector<std::pair<double, double>> setOfPoints) {
    std::vector<std::pair<double, double>> res(setOfPoints);
    size_t n = res.size();
    size_t j;

    for (size_t i = 2; i < n; i++) {
        j = i;

        while (j > 1 && (rotate(res[0], res[j - 1], res[j])) < 0) {
            std::swap(res[j], res[j - 1]);
            j--;
        }
    }

    return res;
}

std::vector<std::pair<double, double>>
grahamScan(std::vector<std::pair<double, double>> setOfPoints) {
    // 1 - Find the leftmost point
    size_t leftmostPoint = getLeftmostPoint(setOfPoints);
    std::swap(setOfPoints[0], setOfPoints[leftmostPoint]);

    // 2 - Sorting all points to the left relative to the starting point
    auto sortingPoints = sortPoints(setOfPoints);

    // 3 - Cut corners
    std::stack<std::pair<double, double>> stackOfPoints;
    stackOfPoints.push(sortingPoints[0]);
    stackOfPoints.push(sortingPoints[1]);

    size_t n = sortingPoints.size();
    size_t stackN = stackOfPoints.size();

    std::pair<double, double> A, B, C;

    for (size_t i = 2; i < n; i++) {
        stackN = stackOfPoints.size();
        B = stackOfPoints.top();
        stackOfPoints.pop();
        A = stackOfPoints.top();
        C = sortingPoints[i];

        if (rotate(A, B, C) > 0) {
            stackOfPoints.push(B);
            stackOfPoints.push(C);
        } else if (stackN < 3) {
            stackOfPoints.push(C);
        } else {
            i--;
        }
    }

    std::vector<std::pair<double, double>> convexHull(stackOfPoints.size());
    size_t iterator = stackOfPoints.size() - 1;
    while (!stackOfPoints.empty()) {
        convexHull[iterator] = stackOfPoints.top();
        stackOfPoints.pop();
        iterator--;
    }

    return convexHull;
}
