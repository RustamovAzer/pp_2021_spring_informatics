// Copyright 2021 Skripal Andrey
#ifndef MODULES_TASK_2_SKRIPAL_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
#define MODULES_TASK_2_SKRIPAL_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
#include <vector>

std::vector<int> getGraph(int size);
std::vector<int> seqDijkstras(std::vector<int> graph, int start, int end);
std::vector<int> parallelDijkstras(std::vector<int> graph, int start, int end);

#endif  // MODULES_TASK_2_SKRIPAL_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
