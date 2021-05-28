// Copyright 2021 Tkachev Alexey
#include <omp.h>
#include <random>
#include <ctime>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <climits>
#include <utility>
#include <iostream>

#include "../../../modules/task_2/tkachev_a_graham_pass/graham_pass.h"

std::vector<Point> getRandomizedVector(double from,
                        double up_to, uint32_t count) {
    assert(from < up_to);
    assert(count > 0);
    std::mt19937 random_generator(time(NULL));
    std::uniform_real_distribution<double> real_dist(from, up_to);
    std::vector<Point> randomized_vector(count);

    for (size_t i = 0; i < count; i++) {
        randomized_vector[i].x = real_dist(random_generator);
        randomized_vector[i].y = real_dist(random_generator);
    }

    return randomized_vector;
}

void swapPoints(Point* p1, Point* p2) {
    std::swap(p1->x, p2->x);
    std::swap(p1->y, p2->y);
}

Point nextToTopOfStack(std::stack<Point> points) {
    assert(!points.empty());
    const Point tmp = points.top();
    points.pop();
    const Point result = points.top();
    points.push(tmp);
    return result;
}

double angleThreePoints(Point p1, Point p2, Point p3) {
     return (p2.x - p1.x) * (p3.y - p1.y) -
            (p2.y - p1.y) * (p3.x - p1.x);
}

double distanceTwoPoints(Point p1, Point p2) {
    return sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                (p2.y - p1.y) * (p2.y - p1.y));
}

std::vector<Point> sortedByPolarAngle(
            std::vector<Point> points, bool multithreading) {
    assert(!points.empty());
    if (multithreading) {
        #pragma omp parallel
        {
            int chunk;
            if (points.size() < 100) {
                chunk = ceil(points.size() / omp_get_num_threads());
            } else {
                chunk = (points.size() / omp_get_num_threads()) / 10;
            }
            for (size_t repeat = 2; repeat < points.size(); repeat++) {
                #pragma omp for ordered schedule(static, chunk)
                for (int j = 2; j < static_cast<int>(points.size()); j++) {
                    double current_angle = angleThreePoints(
                                points[j-1], points[j], points[0]);
                    if (current_angle <= 0) {
                        #pragma omp ordered
                        {
                            swapPoints(&points[j-1], &points[j]);
                        }
                    } else {
                        if (current_angle == 0) {
                            if (distanceTwoPoints(points[j], points[0]) <
                                distanceTwoPoints(points[j-1], points[0])) {
                                #pragma omp ordered
                                {
                                    swapPoints(&points[j-1], &points[j]);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 2; i < points.size(); i++) {
            for (size_t j = 2; j < points.size(); j++) {
                double current_angle = angleThreePoints(
                                    points[j-1], points[j], points[0]);
                if (current_angle <= 0) {
                    swapPoints(&points[j-1], &points[j]);
                } else {
                    if (current_angle == 0) {
                        if (distanceTwoPoints(points[j], points[0]) <
                            distanceTwoPoints(points[j-1], points[0])) {
                            swapPoints(&points[j-1], &points[j]);
                        }
                    }
                }
            }
        }
    }
    return points;
}

uint32_t getIndexMinLeftDownPoint(
                std::vector<Point> points, bool multithreading) {
    assert(!points.empty());

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();

    std::vector<double> min_xs(100, std::numeric_limits<double>::max()),
                        min_ys(100, std::numeric_limits<double>::max());

    if (multithreading) {
        int chunk;
        if (points.size() < 100) {
            chunk = ceil(points.size() / omp_get_num_threads());
        } else {
            chunk = (points.size() / omp_get_num_threads()) / 10;
        }
        #pragma omp parallel firstprivate(points)
        {
            #pragma omp for schedule(static, chunk)
            for (int i = 0; i < static_cast<int>(points.size()); i++) {
                if (min_ys[omp_get_thread_num()] > points[i].y) {
                    min_ys[omp_get_thread_num()] = points[i].y;
                }
            }
        }
        min_y = *std::min_element(min_ys.begin(), min_ys.end());
        #pragma omp parallel firstprivate(points)
        {
            #pragma omp for schedule(static, chunk)
            for (int i = 0; i < static_cast<int>(points.size()); i++) {
                if (points[i].y == min_y &&
                        min_xs[omp_get_thread_num()] > points[i].x) {
                    min_xs[omp_get_thread_num()] = points[i].x;
                }
            }
        }
        min_x = *std::min_element(min_xs.begin(), min_xs.end());
    } else {
        for (size_t i = 0; i < points.size(); i++) {
            if (points[i].y < min_y) {
                min_y = points[i].y;
            }
        }
        for (size_t i = 0; i < points.size(); i++) {
            if (points[i].y == min_y) {
                if (points[i].x < min_x) {
                    min_x = points[i].x;
                }
            }
        }
    }
    for (size_t i = 0; i < points.size(); i++) {
        if (points[i].x == min_x && points[i].y == min_y) {
            return i;
        }
    }
    throw;
}

std::stack<Point> useGrahamAlgorithm(
            std::vector<Point> points, bool multithreading) {
    assert(!points.empty());
    std::stack<Point> points_stack;
    if (points.size() <= 3) {
        for (size_t i = 0; i < points.size(); i++) {
            points_stack.push(points[i]);
        }
        points_stack.push(points[0]);
        return points_stack;
    }
    swapPoints(&points[0],
        &points[getIndexMinLeftDownPoint(points, multithreading)]);

    points = sortedByPolarAngle(points, multithreading);

    points.push_back(points[0]);

    points_stack.push(points[0]);
    points_stack.push(points[1]);
    points_stack.push(points[2]);

    for (size_t i = 3; i < points.size(); i++) {
        while (
            angleThreePoints(
                nextToTopOfStack(points_stack),
                points_stack.top(),
                points[i]) <= 0) {
            if (!points_stack.empty()) {
                points_stack.pop();
            }
        }
        points_stack.push(points[i]);
    }
    return points_stack;
}
