// Copyright 2021 Bakaeva Maria
#include <tbb/tbb.h>
#include <random>
#include <ctime>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <stack>

#include "../../../modules/task_3/bakaeva_m_convex_hull_graham/convex_hull_graham.h"

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

std::vector<std::pair<double, double>>
grahamScan(std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e) {

    std::vector<std::pair<double, double> > setOfPoints(e - b);
    std::copy(b, e, setOfPoints.begin());

    // 1 - Find the leftmost point (omp)
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

std::vector<std::pair<double, double> > grahamScanParallel(
    std::vector<std::pair<double, double> >::iterator b,
    std::vector<std::pair<double, double> >::iterator e,
    int numberOfThreads) {

    if (numberOfThreads == 0) {
        throw "Number of threads is incorrect";
    }

    int st = static_cast<int>((e - b) / numberOfThreads);
    std::vector<std::pair<double, double> > lastPoints;

    // An instance of the class to create threads
    tbb::task_scheduler_init init(numberOfThreads);
    tbb::spin_mutex mutex;
    tbb::task_group g;

    for (int i = 0; i < numberOfThreads - 1; i++) {
        // Create a task
        g.run([&lastPoints, &mutex, i, b, st] () {
            auto l = b + st * i;
            auto r = b + st * (i + 1);
            auto localHull = grahamScan(l, r);

            for (size_t j = 0; j < localHull.size(); j++) {
                // Create capture object
                tbb::spin_mutex::scoped_lock lock;
                // Grab a mutex
                lock.acquire(mutex);

                lastPoints.push_back(localHull[j]);
                // Release mutex
                lock.release();
            }
        });
    }

    g.run([&lastPoints, b, e, st, numberOfThreads, &mutex]() {
        auto localHull = grahamScan(b + st * (numberOfThreads - 1), e);

        for (size_t i = 0; i < localHull.size(); i++) {
            // Create capture object
            tbb::spin_mutex::scoped_lock lock;
            // Grab a mutex
            lock.acquire(mutex);

            lastPoints.push_back(localHull[i]);
            // Release mutex
            lock.release();
        }
    });

    // Wait for task to complete
    g.wait();

    return grahamScan(lastPoints.begin(), lastPoints.end());
}
