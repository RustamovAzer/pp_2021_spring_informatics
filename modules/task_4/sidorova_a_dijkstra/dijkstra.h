// Copyright 2021 Sidorova Alexandra
#ifndef MODULES_TASK_4_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_
#define MODULES_TASK_4_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_

#include <vector>
#include <utility>
#include <iostream>

typedef std::pair<size_t, size_t> Vertex;

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
size_t getVertexWithMinDistanceSeq(const std::vector<std::pair<size_t, bool>>& vertices);

#endif  // MODULES_TASK_4_SIDOROVA_A_DIJKSTRA_DIJKSTRA_H_
