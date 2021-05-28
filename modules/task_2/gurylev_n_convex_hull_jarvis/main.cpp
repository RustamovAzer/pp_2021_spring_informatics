// Copyright 2021 Gurylev Nikita
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <utility>
#include "./convex_hull_jarvis.h"

TEST(ConvexHull, DISABLED_test1) {
    double t1, t2;
    double seqTime, ompTime;
    std::vector<std::pair<int, int>> points = getRandomPoint(500000);
    t1 = omp_get_wtime();
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    t2 = omp_get_wtime();
    seqTime = t2 - t1;
    std::cout << "seqTime = " << seqTime << std::endl;
    t1 = omp_get_wtime();
    std::vector<std::pair<int, int>> alg_omp = JarvisAlgOmp(points);
    t2 = omp_get_wtime();
    ompTime = t2 - t1;
    std::cout << "ompTime = " << ompTime << std::endl;
    ASSERT_EQ(alg_seq, alg_omp);
}

TEST(ConvexHull, DISABLED_test2) {
    double t1, t2;
    double seqTime, ompTime;
    std::vector<std::pair<int, int>> points = getRandomPoint(1000000);
    t1 = omp_get_wtime();
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    t2 = omp_get_wtime();
    seqTime = t2 - t1;
    std::cout << "seqTime = " << seqTime << std::endl;
    t1 = omp_get_wtime();
    std::vector<std::pair<int, int>> alg_omp = JarvisAlgOmp(points);
    t2 = omp_get_wtime();
    ompTime = t2 - t1;
    std::cout << "ompTime = " << ompTime << std::endl;
    ASSERT_EQ(alg_seq, alg_omp);
}

TEST(ConvexHull, test3) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(3, 3), std::pair<int, int>(6, 6),
        std::pair<int, int>(7, 7), std::pair<int, int>(9, 9) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_omp = JarvisAlgOmp(points);
    ASSERT_EQ(alg_seq, alg_omp);
}

TEST(ConvexHull, test4) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(34, 26), std::pair<int, int>(21, 39),
        std::pair<int, int>(0, 50), std::pair<int, int>(4, 15),
        std::pair<int, int>(36, 41), std::pair<int, int>(16, 35),
        std::pair<int, int>(8, 27), std::pair<int, int>(50, 50),
        std::pair<int, int>(50, 0) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_omp = JarvisAlgOmp(points);
    ASSERT_EQ(alg_seq, alg_omp);
}

TEST(ConvexHull, test5) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(3, 8),
        std::pair<int, int>(60, 11), std::pair<int, int>(90, 40),
        std::pair<int, int>(60, 55),
        std::pair<int, int>(13, 33) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_omp = JarvisAlgOmp(points);
    ASSERT_EQ(alg_seq, alg_omp);
}
