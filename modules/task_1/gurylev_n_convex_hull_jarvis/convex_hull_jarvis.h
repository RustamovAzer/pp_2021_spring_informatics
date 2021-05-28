// Copyright 2021 Gurylev Nikita
#ifndef MODULES_TASK_1_GURYLEV_N_CONVEX_HULL_JARVIS_CONVEX_HULL_JARVIS_H_
#define MODULES_TASK_1_GURYLEV_N_CONVEX_HULL_JARVIS_CONVEX_HULL_JARVIS_H_

#include <vector>
#include <utility>

std::vector<std::pair<int, int>> getRandomPoint(int n);
std::pair<int, int> SLeftMPoint(const std::vector<std::pair<int, int>>& points);
bool Dist(const std::pair<int, int> current,
    const std::pair<int, int> next, const std::pair<int, int> temp);
double Around(const std::pair<int, int>& A, const std::pair<int, int>& B,
    const std::pair<int, int>& C);
std::vector < std::pair<int, int>> JarvisAlg(const std::vector<std::pair<int,
    int>>& points);

#endif  // MODULES_TASK_1_GURYLEV_N_CONVEX_HULL_JARVIS_CONVEX_HULL_JARVIS_H_
