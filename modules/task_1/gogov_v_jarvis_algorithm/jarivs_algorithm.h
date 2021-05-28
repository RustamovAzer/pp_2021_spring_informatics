// Copyright 2021 Gogov Vladislav
#ifndef MODULES_TASK_1_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_
#define MODULES_TASK_1_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_

#include <ostream>
#include <vector>
#include <stack>

struct Point {
    int x;
    int y;

    Point() = default;
    Point(int x_, int y_);
    Point(const Point& temp);
    ~Point() = default;

    bool operator<(const Point& temp) const;
    bool operator>(const Point& temp) const;
    Point& operator=(const Point& temp);
    bool operator==(const Point& temp) const;
    bool operator!=(const Point& temp) const;

    friend std::ostream& operator<<(std::ostream& out, const Point& temp);
};

Point searchBottomLeft(const std::vector<Point>& points);
int rotate(const Point& A, const Point& B, const Point& C);
bool checkPointsDistances(const Point& current, const Point& next, const Point& temp);
std::vector<Point> getRandomPoints(int size);
std::vector<Point> jarvisAlgorithm(const std::vector<Point>& points);

#endif  // MODULES_TASK_1_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_
