// Copyright 2021 Gapon Andrey
#include <ctime>
#include <vector>
#include <random>
#include <limits>
#include <utility>
#include <ostream>
#include "../../../modules/task_1/gapon_a_algorithm_dijkstra/dijkstra.h"

#define MAX_DISTANCE 50

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
    if (vertexCount == 0)
        throw "[ERROR] Vertex count must be positive!";

    if (edgeCount > (vertexCount * (vertexCount - 1)) / 2)
        throw "[ERROR] Edge count must be positive and less than (n * (n - 1)) / 2 where 'n' is vertex count!";

    _vertexCount = vertexCount;
    _edgeCount = edgeCount;

    generate();
}

void Graph::generate() {
    std::mt19937 gen;
    gen.seed(static_cast<size_t>(time(0)));

    _distanceMatrix.resize(_vertexCount);
    for (auto& v : _distanceMatrix)
        v.resize(_vertexCount, 0);

    int tempEdgeCount = _edgeCount;
    while (tempEdgeCount > 0) {
        const size_t firstVertex = gen() % _vertexCount;
        const size_t secondVertex = gen() % _vertexCount;
        if (firstVertex == secondVertex || _distanceMatrix[firstVertex][secondVertex] != 0)
            continue;

        _distanceMatrix[firstVertex][secondVertex] =
                _distanceMatrix[secondVertex][firstVertex] = gen() % MAX_DISTANCE + 1;
        tempEdgeCount--;
    }
}

std::vector<std::vector<size_t>> Graph::getDistanceMatrix() const {
    return _distanceMatrix;
}

size_t Graph::getVertexCount() const {
    return _vertexCount;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
    if (graph._vertexCount > 10) {
        out << "Graph is large for printing!" << std::endl;
    } else {
        for (size_t i = 0; i < graph._vertexCount; ++i) {
            for (size_t j = 0; j < i; j++)
                out << "\t";
            for (size_t j = i; j < graph._vertexCount; ++j)
                out << graph._distanceMatrix[j][i] << "\t";
            out << std::endl;
        }
    }

    return out;
}

std::vector<size_t> algorithmDijkstra(const Graph& graph, const size_t root) {
    const auto distanceMatrix = graph.getDistanceMatrix();
    const size_t vertexCount = graph.getVertexCount();
    if (root >= vertexCount)
        throw "[ERROR] Incorrect start vertex!";

    std::vector<std::pair<size_t, bool>> vertices(vertexCount);
    for (auto& p : vertices) {
        p.first = std::numeric_limits<size_t>::max();
        p.second = false;
    }
    vertices[root].first = 0;

    for (size_t i = 0; i < vertexCount; ++i) {
        size_t currentVertex = getVertexWithMinDistance(vertices);
        if (currentVertex == vertexCount)
            break;

        for (size_t j = 0; j < vertexCount; ++j) {
            if (!vertices[j].second && distanceMatrix[currentVertex][j] != 0 &&
                (vertices[currentVertex].first + distanceMatrix[currentVertex][j] < vertices[j].first))
                vertices[j].first = vertices[currentVertex].first + distanceMatrix[currentVertex][j];
        }
        vertices[currentVertex].second = true;
    }

    std::vector<size_t> distances;
    for (auto& p : vertices)
        distances.push_back(p.first);
    return distances;
}

size_t getVertexWithMinDistance(const std::vector<std::pair<size_t, bool>>& vertices) {
    const size_t vertexCount = vertices.size();
    size_t vertex = vertexCount;
    size_t minDistance = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < vertexCount; ++i) {
        if (!vertices[i].second && vertices[i].first < minDistance) {
            minDistance = vertices[i].first;
            vertex = i;
        }
    }

    return vertex;
}

void printDistances(const std::vector<size_t>& distances, const size_t root) {
    std::cout << "Distances for " << root << " vertex: ";
    for (auto d : distances)
        std::cout << d << " ";
    std::cout << std::endl;
}
