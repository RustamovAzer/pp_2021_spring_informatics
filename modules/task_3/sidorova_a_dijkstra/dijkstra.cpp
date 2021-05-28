// Copyright 2021 Sidorova Alexandra
#include <ctime>
#include <vector>
#include <random>
#include <climits>
#include <utility>
#include <ostream>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#include "../../../modules/task_3/sidorova_a_dijkstra/dijkstra.h"

#define MAX_DISTANCE 50
#define THREADS 4

Graph::Graph(const std::vector<std::vector<size_t>>& distanceMatrix) {
    const size_t vertexCount = distanceMatrix.size();
    size_t edgeCount = 0;

    for (size_t i = 0; i < vertexCount; ++i) {
        if (distanceMatrix[i].size() != vertexCount)
            throw "[ERROR] Incorrect distance matrix: different vertex count!";

        for (size_t j = 0; j < vertexCount; ++j) {
            if (j == i && distanceMatrix[i][j] != 0)
                throw "[ERROR] Incorrect distance matrix: graph must be without loops!";
            else if (distanceMatrix[i][j] != 0)
                ++edgeCount;
        }
    }

    _vertexCount = vertexCount;
    _edgeCount = edgeCount;
    _distanceMatrix = distanceMatrix;
}

Graph::Graph(const size_t vertexCount, const size_t edgeCount) {
    if (vertexCount == 0) throw "[ERROR] Vertex count must be positive!";

    if (edgeCount > (vertexCount * (vertexCount - 1)) / 2)
        throw "[ERROR] Edge count must be positive and less than (n * (n - 1)) / 2 where 'n' is vertex count!";

    _vertexCount = vertexCount;
    _edgeCount = edgeCount;

    generate();
}

void Graph::generate() {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    _distanceMatrix.resize(_vertexCount);
    for (auto& v : _distanceMatrix) v.resize(_vertexCount, 0);

    auto generateWeight = [&](const size_t i, const size_t j) {
        _distanceMatrix[i][j] = _distanceMatrix[j][i] = gen() % MAX_DISTANCE + 1;
    };

    if (_edgeCount == (_vertexCount * (_vertexCount - 1) / 2)) {
        for (size_t i = 0; i < _vertexCount; ++i)
            for (size_t j = i + 1; i < _vertexCount; ++i) generateWeight(i, j);
    } else {
        int tempEdgeCount = static_cast<int>(_edgeCount);
        while (tempEdgeCount > 0) {
            const size_t firstVertex = gen() % _vertexCount;
            const size_t secondVertex = gen() % _vertexCount;
            if (firstVertex == secondVertex ||
                _distanceMatrix[firstVertex][secondVertex] != 0)
                continue;

            generateWeight(firstVertex, secondVertex);
            tempEdgeCount--;
        }
    }
}

std::vector<std::vector<size_t>> Graph::getDistanceMatrix() const {
    return _distanceMatrix;
}

size_t Graph::getVertexCount() const { return _vertexCount; }

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
    if (graph._vertexCount > 10) {
        out << "Graph is large for printing!" << std::endl;
    } else {
        for (size_t i = 0; i < graph._vertexCount; ++i) {
            for (size_t j = 0; j < i; j++) out << "\t";
            for (size_t j = i; j < graph._vertexCount; ++j)
                out << graph._distanceMatrix[j][i] << "\t";
            out << std::endl;
        }
    }

    return out;
}

std::vector<size_t> algorithmDijkstraParallel(const Graph& graph, const size_t root) {
    const auto distanceMatrix = graph.getDistanceMatrix();
    const size_t vertexCount = graph.getVertexCount();
    if (root >= vertexCount) throw "[ERROR] Incorrect start vertex!";

    size_t currentVertex = 0;
    std::vector<size_t> distances(vertexCount);
    std::pair<size_t, bool> vertex(UINT_MAX, false);
    std::vector<std::pair<size_t, bool>> vertices(vertexCount, vertex);
    vertices[root].first = 0;

    tbb::task_scheduler_init init(THREADS);

    for (size_t i = 0; i < vertexCount - 1; ++i) {
        currentVertex = getVertexWithMinDistanceParallel(vertices);
        if (currentVertex == vertexCount) break;

        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, vertexCount, vertexCount / THREADS),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t j = range.begin(); j != range.end(); ++j) {
                        if (!vertices[j].second && distanceMatrix[currentVertex][j] != 0 &&
                            (vertices[currentVertex].first +
                             distanceMatrix[currentVertex][j] <
                             vertices[j].first))
                            vertices[j].first = vertices[currentVertex].first +
                                                distanceMatrix[currentVertex][j];
                    }
                });

        vertices[currentVertex].second = true;
    }

    for (size_t i = 0; i < vertexCount; ++i) distances[i] = vertices[i].first;

    return distances;
}

size_t getVertexWithMinDistanceParallel(const std::vector<std::pair<size_t, bool>>& vertices) {
    const size_t vertexCount = vertices.size();
    std::pair<size_t, size_t> vertexPair(UINT_MAX, vertexCount);

    vertexPair = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, vertexCount, vertexCount / THREADS),
            vertexPair,
            [&](const tbb::blocked_range<size_t>& range,
                std::pair<size_t, size_t> currentVertexPair) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    if (!vertices[i].second &&
                        vertices[i].first < currentVertexPair.first) {
                        currentVertexPair.first = vertices[i].first;
                        currentVertexPair.second = i;
                    }
                }

                return currentVertexPair;
            },
            [](std::pair<size_t, size_t> a, std::pair<size_t, size_t> b) {
                return a < b ? a : b;
            });

    return vertexPair.second;
}

std::vector<size_t> algorithmDijkstraSeq(const Graph& graph, const size_t root) {
    const auto distanceMatrix = graph.getDistanceMatrix();
    const size_t vertexCount = graph.getVertexCount();
    if (root >= vertexCount) throw "[ERROR] Incorrect start vertex!";

    std::pair<size_t, bool> vertex(UINT_MAX, false);
    std::vector<std::pair<size_t, bool>> vertices(vertexCount, vertex);
    vertices[root].first = 0;

    for (size_t i = 0; i < vertexCount - 1; ++i) {
        size_t currentVertex = getVertexWithMinDistanceSeq(vertices);
        if (currentVertex == vertexCount) break;

        for (size_t j = 0; j < vertexCount; ++j) {
            if (distanceMatrix[currentVertex][j] != 0 && !vertices[j].second &&
                (vertices[currentVertex].first + distanceMatrix[currentVertex][j] <
                 vertices[j].first))
                vertices[j].first =
                        vertices[currentVertex].first + distanceMatrix[currentVertex][j];
        }
        vertices[currentVertex].second = true;
    }

    std::vector<size_t> distances;
    for (auto& p : vertices) distances.push_back(p.first);
    return distances;
}

size_t getVertexWithMinDistanceSeq(const std::vector<std::pair<size_t, bool>>& vertices) {
    const size_t vertexCount = vertices.size();
    std::pair<size_t, size_t> vertexPair(UINT_MAX, vertexCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        if (!vertices[i].second && vertices[i].first < vertexPair.first) {
            vertexPair.first = vertices[i].first;
            vertexPair.second = i;
        }
    }

    return vertexPair.second;
}

void printDistances(const std::vector<size_t>& distances, const size_t root) {
    std::cout << "Distances for " << root << " vertex: ";
    for (auto d : distances) std::cout << d << " ";
    std::cout << std::endl;
}
