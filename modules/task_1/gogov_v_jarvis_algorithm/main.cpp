// Copyright 2021 Gogov Vladislav
#include <gtest/gtest.h>
#include "./jarivs_algorithm.h"

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Cannot_Apply_Without_Points) {
    std::vector<Point> points(0);
    ASSERT_ANY_THROW(jarvisAlgorithm(points));
}

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Can_Apply_With_One_Point) {
    std::vector<Point> points = {Point(5, 6)};
    std::vector<Point> convex_hull = jarvisAlgorithm(points);
    ASSERT_EQ(points, convex_hull);
}

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Can_Apply_With_Two_Points) {
    std::vector<Point> points = {Point(2, 5), Point(5, 6)};
    std::vector<Point> convex_hull = jarvisAlgorithm(points);
    ASSERT_EQ(points, convex_hull);
}

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Can_Apply_With_One_Line_Points) {
    std::vector<Point> points = {Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4), Point(5, 5)};
    std::vector<Point> convex_hull = jarvisAlgorithm(points);
    std::vector<Point> result_convex_hull = {Point(1, 1), Point(5, 5)};
    ASSERT_EQ(result_convex_hull, convex_hull);
}

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Square_With_Points) {
    std::vector<Point> points = {Point(0, 0), Point(25, 25), Point(34, 10), Point(0, 40), Point(10, 2), Point(40, 40),
                                Point(15, 15), Point(40, 0), Point(5, 5)};
    std::vector<Point> convex_hull = jarvisAlgorithm(points);
    std::vector<Point> result_convex_hull = {Point(0, 0), Point(40, 0), Point(40, 40), Point(0, 40)};
    ASSERT_EQ(result_convex_hull, convex_hull);
}

TEST(Sequential_Jarvis_Algorithm, jarvisAlgorithm_Can_Apply_All_Point_Convex_Hull) {
    std::vector<Point> points = {Point(2, 5), Point(50, 10), Point(80, 30), Point(50, 50), Point(7, 25)};
    std::vector<Point> convex_hull = jarvisAlgorithm(points);
    std::vector<Point> result_convex_hull = points;
    ASSERT_EQ(result_convex_hull, convex_hull);
}

TEST(Struct_Point, Operator_Less) {
    Point a(1, 6);
    Point b(-1, -5);
    ASSERT_TRUE(b < a);
}

TEST(Struct_Point, Operator_Assignment) {
    Point a(9, 16);
    Point b = a;
    ASSERT_EQ(a, b);
}

TEST(Struct_Point, Operator_Equally) {
    Point a(-5, 2);
    Point b(-5, 2);
    ASSERT_TRUE(a == b);
}
