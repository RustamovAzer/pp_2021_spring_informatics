// Copyright 2021 Gapon Andrey
#include <gtest/gtest.h>
#include <vector>
#include "./dijkstra.h"


TEST(Seq_Dijkstra, Test_DifferentVertexCount) {
    const size_t vertexCount = 5;
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount);
    for (int i = vertexCount - 1; i >= 0; --i)
        distanceMatrix[i].resize(i, 0);

    ASSERT_ANY_THROW(Graph graph(distanceMatrix));
}

TEST(Seq_Dijkstra, Test_IncorrectVertexCount) {
    ASSERT_ANY_THROW(Graph graph(0, 0));
}

TEST(Seq_Dijkstra, Test_GraphWithLoops) {
    const size_t vertexCount = 5;
    const std::vector<size_t> distancesForEachVertex(vertexCount, 10);
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount, distancesForEachVertex);
    ASSERT_ANY_THROW(Graph graph(distanceMatrix));
}

TEST(Seq_Dijkstra, Test_IncorrectEdgeCount) {
    const size_t vertexCount = 5;
    const size_t edgeCount = vertexCount * vertexCount;
    ASSERT_ANY_THROW(Graph graph(vertexCount, edgeCount));
}

TEST(Seq_Dijkstra, Test_CreateGraphViaMatrix) {
    const size_t vertexCount = 5;
    std::vector<size_t> eachVertex(vertexCount, 0);
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount, eachVertex);

    distanceMatrix[1][2] = distanceMatrix[2][1] = 2;
    distanceMatrix[1][4] = distanceMatrix[4][1] = 5;
    distanceMatrix[3][2] = distanceMatrix[2][3] = 1;
    distanceMatrix[0][2] = distanceMatrix[2][0] = 10;
    distanceMatrix[0][3] = distanceMatrix[3][0] = 8;
    distanceMatrix[2][4] = distanceMatrix[4][2] = 2;

    Graph graph(distanceMatrix);
    ASSERT_NO_THROW(std::cout << graph);
}

TEST(Seq_Dijkstra, Test_CreateRandomGraph) {
    const size_t vertexCount = 5;
    const size_t edgeCount = 6;

    Graph graph(vertexCount, edgeCount);
    ASSERT_NO_THROW(std::cout << graph);
}

TEST(Seq_Dijkstra, Test_CreateRandomFullDefaultGraph) {
    Graph graph;
    ASSERT_NO_THROW(std::cout << graph);
}

TEST(Seq_Dijkstra, Test_DijkstraForRandomGraph) {
    const size_t vertexCount = 8;
    const size_t edgeCount = 18;

    Graph graph(vertexCount, edgeCount);
    ASSERT_NO_THROW(std::cout << graph);

    for (size_t i = 0; i < vertexCount; ++i) {
        auto distances = algorithmDijkstra(graph, i);
        printDistances(distances, i);
    }
}

TEST(Seq_Dijkstra, Test_DijkstraForHardcodeGraph) {
    const size_t vertexCount = 5;
    std::vector<size_t> eachVertex(vertexCount, 0);
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount, eachVertex);

    distanceMatrix[1][2] = distanceMatrix[2][1] = 2;
    distanceMatrix[1][4] = distanceMatrix[4][1] = 5;
    distanceMatrix[3][2] = distanceMatrix[2][3] = 1;
    distanceMatrix[0][2] = distanceMatrix[2][0] = 10;
    distanceMatrix[0][3] = distanceMatrix[3][0] = 8;
    distanceMatrix[3][4] = distanceMatrix[4][3] = 1;

    Graph graph(distanceMatrix);
    ASSERT_NO_THROW(std::cout << graph);

    std::vector<std::vector<size_t>> distancesRef = {
            {0, 11, 9, 8, 9}, {11, 0, 2, 3, 4}, {9, 2, 0, 1, 2}, {8, 3, 1, 0, 1}, {9, 4, 2, 1, 0}
    };
    for (size_t i = 0; i < vertexCount; ++i) {
        auto distancesSeq = algorithmDijkstra(graph, i);
        ASSERT_EQ(distancesRef[i], distancesSeq);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
