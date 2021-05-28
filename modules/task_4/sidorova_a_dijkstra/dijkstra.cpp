// Copyright 2021 Sidorova Alexandra
#include <ctime>
#include <vector>
#include <random>
#include <climits>
#include <utility>
#include <ostream>
#include "../../../modules/task_4/sidorova_a_dijkstra/dijkstra.h"
#include "../../../3rdparty/unapproved/unapproved.h"

#define MAX_DISTANCE 50
#define THREADS 1

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

    const Vertex max(UINT_MAX, vertexCount);
    Vertex globalMin(UINT_MAX, vertexCount);
    size_t currentVertex = vertexCount;
    size_t currentDistance = UINT_MAX;
    std::vector<size_t> distances(vertexCount);
    std::vector<Vertex> vertices(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i)
        vertices[i] = std::make_pair(UINT_MAX, i);
    std::vector<bool> marks(vertexCount, false);
    vertices[root].first = 0;

    std::vector<size_t> limits(THREADS + 1, vertexCount);
    const size_t delta = vertexCount / THREADS;
    for (size_t i = 0; i < THREADS; i++)
        limits[i] = i * delta;

    auto getMinDistance = [&](const int ithr) -> const Vertex& {
        std::reference_wrapper<const Vertex> localMin(globalMin);
        for (size_t i = limits[ithr]; i < limits[ithr + 1]; ++i) {
            if (!marks[i] && vertices[i].first < localMin.get().first) {
                localMin = vertices[i];
            }
        }
        return localMin;
    };

    auto writeDistances = [&](const int ithr) {
        for (size_t j = limits[ithr]; j < limits[ithr + 1]; ++j) {
            const size_t distanceToVertex = distanceMatrix[currentVertex][j];
            if (distanceToVertex != 0 && !marks[j] && (currentDistance + distanceToVertex < vertices[j].first))
                vertices[j].first = currentDistance + distanceToVertex;
        }
    };

    std::vector<std::future<const Vertex&>> local_min_distances(THREADS);
    std::vector<std::future<void>> writers(THREADS);
    for (size_t i = 0; i < vertexCount - 1; ++i) {
        globalMin = max;
        for (int j = 0; j < THREADS; ++j)
            local_min_distances[j] = std::async(std::launch::deferred, getMinDistance, j);

        for (auto& min_dist : local_min_distances) {
            auto tmp = min_dist.get();
            if (tmp.first < globalMin.first)
                globalMin = tmp;
        }

        currentVertex = globalMin.second;
        if (currentVertex == vertexCount) break;

        currentDistance = vertices[currentVertex].first;
        marks[currentVertex] = true;

        for (int j = 0; j < THREADS; ++j)
            writers[j] = std::async(std::launch::deferred, writeDistances, j);
        for (auto& w : writers)
            w.get();
    }

    for (size_t i = 0; i < vertexCount; ++i) distances[i] = vertices[i].first;

    return distances;
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
            if (!vertices[j].second && distanceMatrix[currentVertex][j] != 0 &&
                (vertices[currentVertex].first + distanceMatrix[currentVertex][j] < vertices[j].first))
                vertices[j].first = vertices[currentVertex].first + distanceMatrix[currentVertex][j];
        }
        vertices[currentVertex].second = true;
    }

    std::vector<size_t> distances;
    for (auto& p : vertices) distances.push_back(p.first);
    return distances;
}

size_t getVertexWithMinDistanceSeq(const std::vector<std::pair<size_t, bool>>& vertices) {
    const size_t vertexCount = vertices.size();
    Vertex vertexPair(UINT_MAX, vertexCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        if (!vertices[i].second && vertices[i].first < vertexPair.first) {
            vertexPair.first = vertices[i].first;
            vertexPair.second = i;
        }
    }

    return vertexPair.second;
}
