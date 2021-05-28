// Copyright 2021 Skripal Andrey
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include <vector>
#include <iostream>
#include "../../../modules/task_3/skripal_a_dijkstra_algorithm/dijkstra_algorithm.h"


TEST(Deikstra_Algorithm, test1) {
    int size = 72;
    int start = 1;
    int end = 70;
    tbb::tick_count t1, t2;
    std::vector<int> graph = getGraph(size);

    t1 = tbb::tick_count::now();
    std::vector<int> res1 = parallelDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "parallel Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> res2 = seqDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "seq Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, test2) {
    int size = 80;
    int start = 2;
    int end = 66;
    tbb::tick_count t1, t2;
    std::vector<int> graph = getGraph(size);

    t1 = tbb::tick_count::now();
    std::vector<int> res1 = parallelDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "parallel Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> res2 = seqDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "seq Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, test3) {
    int size = 90;
    int start = 4;
    int end = 89;
    tbb::tick_count t1, t2;
    std::vector<int> graph = getGraph(size);

    t1 = tbb::tick_count::now();
    std::vector<int> res1 = parallelDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "parallel Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> res2 = seqDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "seq Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, test4) {
    int size = 50;
    int start = 12;
    int end = 49;
    tbb::tick_count t1, t2;
    std::vector<int> graph = getGraph(size);

    t1 = tbb::tick_count::now();
    std::vector<int> res1 = parallelDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "parallel Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> res2 = seqDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "seq Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(res1, res2);
}

TEST(Deikstra_Algorithm, test5) {
    int size = 120;
    int start = 10;
    int end = 111;
    tbb::tick_count t1, t2;
    std::vector<int> graph = getGraph(size);

    t1 = tbb::tick_count::now();
    std::vector<int> res1 = parallelDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "parallel Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> res2 = seqDijkstras(graph, start, end);
    t2 = tbb::tick_count::now();
    std::cout << "seq Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(res1, res2);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
