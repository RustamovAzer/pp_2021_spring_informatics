// Copyright 2021 Sidorova Alexandra
#include <gtest/gtest.h>
#include <vector>
#include <ostream>
#include <climits>
#include "./dijkstra.h"
#include "../../../3rdparty/unapproved/unapproved.h"


TEST(STD_Dijkstra, DISABLED_Test_Time) {
    const size_t root = 0;
    const size_t vertexCount = 10000;
    const size_t edgeCount = vertexCount * (vertexCount - 1) / 2;

    Graph graph(vertexCount, edgeCount);

    auto begin = std::chrono::high_resolution_clock::now();
    const auto distancesSeq = algorithmDijkstraSeq(graph, root);
    auto end = std::chrono::high_resolution_clock::now();
    const auto seq = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Sequential Time: " << seq << std::endl;

    begin = std::chrono::high_resolution_clock::now();
    const auto distancesParallel = algorithmDijkstraParallel(graph, root);
    end = std::chrono::high_resolution_clock::now();
    const auto par = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Parallel Time: " << par << std::endl;

    std::cout << (static_cast<float>(seq - par) / (static_cast<float>(par))) << std::endl;

    ASSERT_EQ(distancesSeq, distancesParallel);
}

TEST(STD_Dijkstra, Test_DifferentVertexCount) {
    const size_t vertexCount = 5;
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount);
    for (int i = vertexCount - 1; i >= 0; --i)
        distanceMatrix[i].resize(i, 0);

    ASSERT_ANY_THROW(Graph graph(distanceMatrix));
}

TEST(STD_Dijkstra, Test_IncorrectVertexCount) {
    ASSERT_ANY_THROW(Graph graph(0, 0));
}

TEST(STD_Dijkstra, Test_GraphWithLoops) {
    const size_t vertexCount = 5;
    const std::vector<size_t> distancesForEachVertex(vertexCount, 10);
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount, distancesForEachVertex);
    ASSERT_ANY_THROW(Graph graph(distanceMatrix));
}

TEST(STD_Dijkstra, Test_IncorrectEdgeCount) {
    const size_t vertexCount = 5;
    const size_t edgeCount = vertexCount * vertexCount;
    ASSERT_ANY_THROW(Graph graph(vertexCount, edgeCount));
}

TEST(STD_Dijkstra, Test_CreateGraphViaMatrix) {
    const size_t vertexCount = 5;
    std::vector<size_t> eachVertex(vertexCount, 0);
    std::vector<std::vector<size_t>> distanceMatrix(vertexCount, eachVertex);

    distanceMatrix[1][2] = distanceMatrix[2][1] = 2;
    distanceMatrix[1][4] = distanceMatrix[4][1] = 5;
    distanceMatrix[3][2] = distanceMatrix[2][3] = 1;
    distanceMatrix[0][2] = distanceMatrix[2][0] = 10;
    distanceMatrix[0][3] = distanceMatrix[3][0] = 8;
    distanceMatrix[2][4] = distanceMatrix[4][2] = 2;

    ASSERT_NO_THROW(Graph graph(distanceMatrix));
}

TEST(STD_Dijkstra, Test_CreateRandomGraph) {
    const size_t vertexCount = 5;
    const size_t edgeCount = 6;

    ASSERT_NO_THROW(Graph graph(vertexCount, edgeCount));
}

TEST(STD_Dijkstra, Test_CreateRandomFullDefaultGraph) {
    ASSERT_NO_THROW(Graph graph);
}

TEST(STD_Dijkstra, Test_DijkstraForHardcodeGraph) {
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

    const std::vector<std::vector<size_t>> distancesRef = {
            {0, 11, 9, 8, 9}, {11, 0, 2, 3, 4}, {9, 2, 0, 1, 2}, {8, 3, 1, 0, 1}, {9, 4, 2, 1, 0}
    };

    for (size_t i = 0; i < vertexCount; ++i) {
        auto distancesSeq = algorithmDijkstraSeq(graph, i);
        auto distancesParallel = algorithmDijkstraParallel(graph, i);
        ASSERT_EQ(distancesRef[i], distancesSeq);
        ASSERT_EQ(distancesRef[i], distancesParallel);
    }
}

TEST(STD_Dijkstra, Test_DijkstraSeqForRandomGraph) {
    const size_t vertexCount = 8;
    const size_t edgeCount = 18;

    Graph graph(vertexCount, edgeCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        ASSERT_NO_THROW(algorithmDijkstraSeq(graph, i));
    }
}

TEST(STD_Dijkstra, Test_DijkstraForRandomGraph) {
    const size_t vertexCount = 10;
    const size_t edgeCount = vertexCount * (vertexCount - 1) / 4;

    Graph graph(vertexCount, edgeCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        auto distancesSeq = algorithmDijkstraSeq(graph, i);
        auto distancesParallel = algorithmDijkstraParallel(graph, i);
        ASSERT_EQ(distancesSeq, distancesParallel);
    }
}

TEST(STD_Dijkstra, Test_DijkstraForRandomFullGraph) {
    const size_t vertexCount = 10;
    const size_t edgeCount = vertexCount * (vertexCount - 1) / 2;

    Graph graph(vertexCount, edgeCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        auto distancesSeq = algorithmDijkstraSeq(graph, i);
        auto distancesParallel = algorithmDijkstraParallel(graph, i);
        ASSERT_EQ(distancesSeq, distancesParallel);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
