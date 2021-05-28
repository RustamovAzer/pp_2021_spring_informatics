// Copyright 2021 Bakaeva Maria
#include <omp.h>
#include <random>
#include <ctime>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <stack>

#include "../../../modules/task_2/bakaeva_m_convex_hull_graham/convex_hull_graham.h"

std::vector<std::pair<double, double>> generateRandomPoints(size_t count) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<double> dis(50, 550);

    std::vector<std::pair<double, double> > points(count);
    for (std::size_t i = 0; i < count; ++i) {
        points[i] = std::make_pair(dis(gen), dis(gen));
    }
    return points;
}

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

size_t getLeftmostPointOmp
(std::vector < std::pair<double, double>> setOfPoints) {
    size_t n = setOfPoints.size();
    size_t leftmostPoint = 0;
    size_t commonLeftmostPoint = 0;

#pragma omp parallel firstprivate(leftmostPoint)
    {
#pragma omp for
        for (int i = 0; i < static_cast<int>(n); i++) {
            if (setOfPoints[i] < setOfPoints[leftmostPoint]) {
                leftmostPoint = i;
            }
        }

#pragma omp critical
        if (setOfPoints[leftmostPoint] < setOfPoints[commonLeftmostPoint]) {
            commonLeftmostPoint = leftmostPoint;
        }
    }

    return commonLeftmostPoint;
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

std::vector<std::pair<double, double>>
grahamScanOmp(std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e) {

    std::vector<std::pair<double, double> > setOfPoints(e - b);
    std::copy(b, e, setOfPoints.begin());

    // 1 - Find the leftmost point (omp)
    size_t leftmostPoint = getLeftmostPointOmp(setOfPoints);
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

std::vector<std::pair<double, double> > grahamScanParallel(
    std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e,
    int numberOfThreads) {

    if (numberOfThreads == 0) {
        throw "Number of threads is incorrect";
    }

    int st = static_cast<int>((e - b) / numberOfThreads);
    std::vector<std::pair<double, double> > lastPoints;

#pragma omp parallel num_threads(numberOfThreads)
    {
        size_t thread = omp_get_thread_num();
        std::vector<std::pair<double, double> > localHull;

        if (static_cast<int>(thread) == numberOfThreads - 1) {
            localHull = grahamScanOmp(b + st * thread, e);
        } else {
            localHull
                = grahamScanOmp(b + st * thread, b + st * (thread + 1));
        }

#pragma omp critical
        {
            for (size_t i = 0; i < localHull.size(); i++) {
                lastPoints.push_back(localHull[i]);
            }
        }
    }

    return grahamScanOmp(lastPoints.begin(), lastPoints.end());
}
