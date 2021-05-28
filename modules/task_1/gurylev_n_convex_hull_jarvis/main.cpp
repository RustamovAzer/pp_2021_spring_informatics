// Copyright 2021 Gurylev Nikita
#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include "./convex_hull_jarvis.h"

TEST(ConvexHull, test1) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(15, 25) };
    std::vector<std::pair<int, int>> alg = JarvisAlg(points);
    ASSERT_EQ(points, alg);
}

TEST(ConvexHull, test2) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(3, 4),
        std::pair<int, int>(4, 9) };
    std::vector<std::pair<int, int>> alg = JarvisAlg(points);
    ASSERT_EQ(points, alg);
}

TEST(ConvexHull, test3) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(3, 3), std::pair<int, int>(6, 6),
        std::pair<int, int>(7, 7), std::pair<int, int>(9, 9) };
    std::vector<std::pair<int, int>> alg = JarvisAlg(points);
    std::vector<std::pair<int, int>> result = { std::pair<int, int>(0, 0),
        std::pair<int, int>(9, 9) };
    ASSERT_EQ(result, alg);
}

TEST(ConvexHull, test4) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(34, 26), std::pair<int, int>(21, 39),
        std::pair<int, int>(0, 50), std::pair<int, int>(4, 15),
        std::pair<int, int>(36, 41), std::pair<int, int>(16, 35),
        std::pair<int, int>(8, 27), std::pair<int, int>(50, 50),
        std::pair<int, int>(50, 0) };
    std::vector<std::pair<int, int>> alg = JarvisAlg(points);
    std::vector<std::pair<int, int>> result = { std::pair<int, int>(0, 0),
        std::pair<int, int>(50, 0), std::pair<int, int>(50, 50),
        std::pair<int, int>(0, 50) };
    ASSERT_EQ(result, alg);
}

TEST(ConvexHull, test5) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(3, 8),
        std::pair<int, int>(60, 11), std::pair<int, int>(90, 40),
        std::pair<int, int>(60, 55),
        std::pair<int, int>(13, 33) };
    std::vector<std::pair<int, int>> alg = JarvisAlg(points);
    std::vector<std::pair<int, int>> result = points;
    ASSERT_EQ(result, alg);
}
