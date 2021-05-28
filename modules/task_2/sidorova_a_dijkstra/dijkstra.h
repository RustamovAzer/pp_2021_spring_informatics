// Copyright 2021 Sidorova Alexandra
#ifndef MODULES_TASK_2_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_
#define MODULES_TASK_2_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_

#include <vector>
#include <utility>
#include <iostream>

class Graph {
 public:
    explicit Graph(const std::vector<std::vector<size_t>>& distanceMatrix);
    explicit Graph(const size_t vertexCount = 10, const size_t edgeCount = 45);
    ~Graph() = default;

    std::vector<std::vector<size_t>> getDistanceMatrix() const;
    size_t getVertexCount() const;

    friend std::ostream& operator<<(std::ostream& out, const Graph& graph);
 private:
    void generate();

    size_t _vertexCount;
    size_t _edgeCount;
    std::vector<std::vector<size_t>> _distanceMatrix;
};

std::vector<size_t> algorithmDijkstraParallel(const Graph& graph, const size_t root);
std::vector<size_t> algorithmDijkstraSeq(const Graph& graph, const size_t root);
size_t getVertexWithMinDistanceParallel(const std::vector<std::pair<size_t, bool>>& vertices);
size_t getVertexWithMinDistanceSeq(const std::vector<std::pair<size_t, bool>>& vertices);
void printDistances(const std::vector<size_t>& distances, const size_t root);

#endif  // MODULES_TASK_2_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_
