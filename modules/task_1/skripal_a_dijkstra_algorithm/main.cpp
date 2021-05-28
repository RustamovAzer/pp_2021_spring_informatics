// Copyright 2021 Skripal Andrey
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/skripal_a_dijkstra_algorithm/dijkstra_algorithm.h"

TEST(Deikstra_Algorithm, 6_vertex) {
    std::vector<int> graph = { 0, 7, 9, 0, 0, 14,
                               7, 0, 10, 12, 0, 0,
                               9, 10, 0, 11, 0, 2,
                               0, 12, 11, 5, 6, 0,
                               0, 0, 5, 6, 0, 9,
                               14, 0, 2, 0, 9, 0 };

    std::vector<int> res1 = { 5, 6, 3, 1 };
    std::vector<int> res2 = seqDijkstras(graph, 1, 5);
    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, 4_vertex) {
    std::vector<int> graph = { 0, 2, 1, 4,
                               2, 0, 5, 1,
                               1, 5, 0, 7,
                               4, 1, 7, 0 };

    std::vector<int> res1 = { 4, 2, 1 };
    std::vector<int> res2 = seqDijkstras(graph, 1, 4);
    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, 5_vertex) {
    std::vector<int> graph = { 0, 9, 4, 0, 11,
                               9, 0, 5, 7, 2,
                               4, 5, 0, 3, 1,
                               0, 7, 3, 0, 2,
                               11, 2, 1, 2, 0 };

    std::vector<int> res1 = { 4, 5, 2 };
    std::vector<int> res2 = seqDijkstras(graph, 2, 4);
    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, 5_vertex2) {
    std::vector<int> graph = { 0, 9, 4, 0, 11,
                               9, 0, 5, 7, 2,
                               4, 5, 0, 3, 1,
                               0, 7, 3, 0, 2,
                               11, 2, 1, 2, 0 };

    std::vector<int> res1 = { 5, 3, 1 };
    std::vector<int> res2 = seqDijkstras(graph, 1, 5);
    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, 7_vertex) {
    std::vector<int> graph = { 0, 2, 0, 7, 0, 5, 10,
                               2, 0, 0, 1, 4, 9, 0,
                                0, 0, 0, 17, 0, 3, 3,
                                7, 1, 17, 0, 1, 4, 2,
                                0, 4, 0, 1, 0, 6, 0,
                                5, 9, 3, 4, 6, 0, 1,
                                10, 0, 3, 2, 0, 1, 0 };

    std::vector<int> res1 = { 7, 4, 2, 1 };
    std::vector<int> res2 = seqDijkstras(graph, 1, 7);
    ASSERT_EQ(res1, res2);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
