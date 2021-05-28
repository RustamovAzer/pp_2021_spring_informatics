// Copyright 2021 Tkachev Alexey
#ifndef MODULES_TASK_2_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
#define MODULES_TASK_2_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
#include <vector>
#include <stack>
#include <cstdint>

struct Point {double x, y;};

std::vector<Point> getRandomizedVector(
    double from, double up_to, uint32_t count);

double angleThreePoints(Point p1, Point p2, Point p3);
double distanceTwoPoints(Point p1, Point p2);

void swapPoints(Point* p1, Point* p2);
Point nextToTopOfStack(std::stack<Point> points);

std::vector<Point> sortedByPolarAngle(
    std::vector<Point> points, bool multithreading);

uint32_t getIndexMinLeftDownPoint(
    std::vector<Point> points, bool multithreading);

std::stack<Point> useGrahamAlgorithm(
    std::vector<Point> points, bool multithreading);

#endif  // MODULES_TASK_2_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
