// Copyright 2021 Gapon Andrey

#include "../../../modules/task_2/gapon_a_algorithm_dijkstra/dijkstra.h"
#include <omp.h>
#include <vector>
#include <limits>
#include <cmath>

std::vector<int> generateSimpleGraph(int size) {
    std::vector<int> sample(size * size, 2);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                sample[i * size + j] = 1;
            }
        }
    }
    return sample;
}

std::vector<int> dijkstra(const std::vector<int>& graph, int start, int end) {
    if (graph.size() == 0) {
        throw "Empty graph";
    }

    if (start == end)
        return std::vector<int>(1, 0);

    if (start > end) {
        std::swap(start, end);
    }

    int points_count = sqrt(graph.size());

    if (sqrt(graph.size()) != points_count) {
        throw "Wrong size";
    }

    int max_weight = std::numeric_limits<int>::max();
    int min, min_point, tmp;
    std::vector<int> points_len(points_count, max_weight);
    std::vector<int> path;
    std::vector<bool> processed(points_count, false);

    // Align start and end with array indexes
    --start;
    --end;

    points_len[start] = 0;

    do {
        min_point = max_weight;
        min = max_weight;

        // Choose a point to work with
        for (int i = 0; i < points_count; i++) {
            if (!processed[i] && points_len[i] < min) {
                min_point = i;
                min = points_len[i];
            }
        }

        if (min_point != max_weight) {
            for (int i = 0; i < points_count; i++) {
                if (graph[min_point * points_count + i] > 0) {
                    tmp = min + graph[min_point * points_count + i];
                    if (points_len[i] > tmp) {
                        points_len[i] = tmp;
                    }
                }
            }
            processed[min_point] = true;
        }
    } while (min_point < max_weight);

    // Configuring a path
    path.push_back(end + 1);
    int weight = points_len[end];

    while (end != start) {
        for (int i = 0; i < points_count; i++) {
            if (graph[end * points_count + i] < 0) {
                throw "Graph weight can not be less then zero.";
            }
            if (graph[end * points_count + i] > 0) {
                tmp = weight - graph[end * points_count + i];
                if (points_len[i] == tmp) {
                    weight = tmp;
                    end = i;
                    path.push_back(i + 1);
                }
            }
        }
    }

    return path;
}

std::vector<int> dijkstraParallel(const std::vector<int>& graph,
    int start, int end) {
    if (graph.size() == 0) {
        throw "Empty graph";
    }

    if (start == end)
        return std::vector<int>(1, 0);

    if (start > end) {
        std::swap(start, end);
    }

    int points_count = sqrt(graph.size());

    if (sqrt(graph.size()) != points_count) {
        throw "Wrong size";
    }

    int max_weight = std::numeric_limits<int>::max();
    int min_point, min, tmp;
    std::vector<int> points_len(points_count, max_weight);
    std::vector<int> path;
    std::vector<bool> processed(points_count, false);

    // Align start and end with array indexes
    --start;
    --end;

    points_len[start] = 0;
    int retry = 10;
    do {
        min_point = max_weight;
        min = max_weight;


#pragma omp parallel
        {
            int local_min_point = max_weight;
            int local_min = max_weight;

#pragma omp for
            for (int i = 0; i < points_count; i++) {
                if (!processed[i] && points_len[i] < local_min) {
                    local_min = points_len[i];
                    local_min_point = i;
                }
            }

#pragma omp critical
            {
                if (local_min < min) {
                    min = local_min;
                    min_point = local_min_point;
                }
            }
        }

        if (min_point != max_weight) {
#pragma omp parallel for private(tmp)
            for (int i = 0; i < points_count; i++) {
                if (graph[min_point * points_count + i] > 0) {
                    tmp = min + graph[min_point * points_count + i];
                    if (points_len[i] > tmp) {
                        points_len[i] = tmp;
                    }
                }
            }
            processed[min_point] = true;
        }
        retry--;
    } while (min_point < max_weight || retry == 0);

    path.push_back(end + 1);
    int weight = points_len[end];

    while (end != start) {
#pragma omp parallel for private(tmp)
        for (int i = 0; i < points_count; i++) {
            if (graph[end * points_count + i] > 0) {
                tmp = weight - graph[end * points_count + i];
                if (points_len[i] == tmp) {
#pragma omp critical
                    {
                        weight = tmp;
                        end = i;
                        path.push_back(i + 1);
                    }
                }
            }
        }
    }
    return path;
}
