// Copyright 2021 Gapon Andrey
#ifndef MODULES_TASK_2_GAPON_A_ALGORITHM_DIJKSTRA_DIJKSTRA_H_
#define MODULES_TASK_2_GAPON_A_ALGORITHM_DIJKSTRA_DIJKSTRA_H_

#include <vector>

std::vector<int> generateSimpleGraph(int size);
std::vector<int> dijkstra(const std::vector<int>& graph, int start, int end);
std::vector<int> dijkstraParallel(const std::vector<int>& graph,
int start, int end);

#endif  // MODULES_TASK_2_GAPON_A_ALGORITHM_DIJKSTRA_DIJKSTRA_H_
