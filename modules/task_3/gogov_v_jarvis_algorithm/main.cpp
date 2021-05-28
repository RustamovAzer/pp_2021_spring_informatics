// Copyright 2021 Gogov Vladislav
#include <gtest/gtest.h>
#include "tbb/tick_count.h"
#include "./jarivs_algorithm.h"

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Cannot_Apply_Without_Points) {
    std::vector<Point> points(0);
    ASSERT_ANY_THROW(jarvisAlgorithmTbb(points));
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Can_Apply_With_One_Point) {
    std::vector<Point> points{Point(5, 6)};
    std::vector<Point> convex_hull_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hull_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hull_seq, convex_hull_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Can_Apply_With_Two_Points) {
    std::vector<Point> points{Point(2, 5), Point(5, 6)};
    std::vector<Point> convex_hull_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hull_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hull_seq, convex_hull_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Can_Apply_With_One_Line_Points) {
    std::vector<Point> points{Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4), Point(5, 5)};
    std::vector<Point> convex_hull_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hull_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hull_seq, convex_hull_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Square_With_Points) {
    std::vector<Point> points{Point(0, 0),   Point(25, 25), Point(34, 10), Point(0, 40), Point(10, 2),
                              Point(40, 40), Point(15, 15), Point(40, 0),  Point(5, 5)};
    std::vector<Point> convex_hull_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hull_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hull_seq, convex_hull_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_Can_Apply_All_Point_Convex_Hull) {
    std::vector<Point> points{Point(2, 5), Point(50, 10), Point(80, 30), Point(50, 50), Point(7, 25)};
    std::vector<Point> convex_hull_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hull_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hull_seq, convex_hull_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_With_Size_102) {
    size_t count_point = 102ull;
    std::vector<Point> points = getRandomPoints(count_point);
    std::vector<Point> convex_hul_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hul_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hul_seq, convex_hul_tbb);
}

TEST(TBB_Jarvis_Algorithm, Jarvis_Algorithm_With_Size_111) {
    size_t count_point = 111ull;
    std::vector<Point> points = getRandomPoints(count_point);
    std::vector<Point> convex_hul_seq = jarvisAlgorithmSeq(points);
    std::vector<Point> convex_hul_tbb = jarvisAlgorithmTbb(points);
    ASSERT_EQ(convex_hul_seq, convex_hul_tbb);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
