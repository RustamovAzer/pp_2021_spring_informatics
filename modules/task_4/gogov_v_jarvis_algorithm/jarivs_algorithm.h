// Copyright 2021 Gogov Vladislav
#ifndef MODULES_TASK_4_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_
#define MODULES_TASK_4_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_
#include <vector>
#include <utility>

using Point = std::pair<int64_t, int64_t>;

Point searchBottomLeft(const std::vector<Point>& points);
int rotate(const Point& A, const Point& B, const Point& C);
bool checkPointsDistances(const Point& current, const Point& next, const Point& temp);
std::vector<Point> getRandomPoints(size_t size);
std::vector<Point> jarvisAlgorithmSeq(const std::vector<Point>& points);
std::vector<Point> jarvisAlgorithmStd(const std::vector<Point>& points);

#endif  // MODULES_TASK_4_GOGOV_V_JARVIS_ALGORITHM_JARIVS_ALGORITHM_H_
