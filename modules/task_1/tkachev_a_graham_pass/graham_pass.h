// Copyright 2021 Tkachev Alexey
#ifndef MODULES_TASK_1_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
#define MODULES_TASK_1_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
#include <vector>
#include <stack>

struct Point {double x, y;};
std::vector<Point> getRandomizedVector(
    double from, double up_to, uint32_t count);
void swapPoints(Point* p1, Point* p2);
Point nextToTopOfStack(std::stack<Point> points);
double angleThreePoints(Point p1, Point p2, Point p3);
double distanceTwoPoints(Point p1, Point p2);
std::vector<Point> sortedByPolarAngle(std::vector<Point> points);
uint32_t getIndexMinLeftDownPoint(std::vector<Point> points);
std::stack<Point> useGrahamAlgorithm(std::vector<Point> points);

#endif  // MODULES_TASK_1_TKACHEV_A_GRAHAM_PASS_GRAHAM_PASS_H_
