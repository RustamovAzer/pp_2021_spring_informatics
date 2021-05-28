// Copyright 2021 Bakaeva Maria
#include <gtest/gtest.h>
#include <tbb/tbb.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include "./convex_hull_graham.h"

TEST(ConvexHull, DISABLED_timeTest) {
    size_t size = 100000;
    std::vector<std::pair<double, double> > points(size);
    points = generateRandomPoints(size);

    tbb::tick_count t1 = tbb::tick_count::now();
    auto resultSeq = grahamScan(points.begin(), points.end());
    tbb::tick_count t2 = tbb::tick_count::now();

    std::cout << " Sequential time " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    auto resultParallel = grahamScanParallel(points.begin(), points.end(), 4);
    t2 = tbb::tick_count::now();

    std::cout << " Parallel time " << (t2 - t1).seconds() << std::endl;

    for (size_t i = 0; i < resultParallel.size(); i++) {
        ASSERT_NEAR(resultParallel[i].first, resultSeq[i].first, 0.00001);
        ASSERT_NEAR(resultParallel[i].second, resultSeq[i].second, 0.00001);
    }
}

TEST(ConvexHull, exceptionThreadsTest) {
  std::vector<std::pair<double, double> > result(12);
  result[0] = std::make_pair(-1, 2);
  result[1] = std::make_pair(-2, 6);
  result[2] = std::make_pair(1, 7);
  result[3] = std::make_pair(-4, 0);
  result[4] = std::make_pair(2, 4);
  result[5] = std::make_pair(5, 5);
  result[6] = std::make_pair(-1, -1);
  result[7] = std::make_pair(1, 1);
  result[8] = std::make_pair(3, 3);
  result[9] = std::make_pair(1, -3);
  result[10] = std::make_pair(4, -2);
  result[11] = std::make_pair(4, 1);

  ASSERT_ANY_THROW(grahamScanParallel(result.begin(), result.end(), 0));
}

TEST(ConvexHull, getConvexHullGrahamTbb_1) {
  std::vector<std::pair<double, double> > result(12);
  result[0] = std::make_pair(-1, 2);
  result[1] = std::make_pair(-2, 6);
  result[2] = std::make_pair(1, 7);
  result[3] = std::make_pair(-4, 0);
  result[4] = std::make_pair(2, 4);
  result[5] = std::make_pair(5, 5);
  result[6] = std::make_pair(-1, -1);
  result[7] = std::make_pair(1, 1);
  result[8] = std::make_pair(3, 3);
  result[9] = std::make_pair(1, -3);
  result[10] = std::make_pair(4, -2);
  result[11] = std::make_pair(4, 1);

  result = grahamScanParallel(result.begin(), result.end(), 4);

  std::vector<std::pair<double, double> > check(result.size());
  check[0] = std::make_pair(-4, 0);
  check[1] = std::make_pair(1, -3);
  check[2] = std::make_pair(4, -2);
  check[3] = std::make_pair(5, 5);
  check[4] = std::make_pair(1, 7);
  check[5] = std::make_pair(-2, 6);

  ASSERT_EQ(result, check);
}

TEST(ConvexHull, getConvexHullGrahamTbb_2) {
  std::vector<std::pair<double, double> > result(20);
  result[0] = std::make_pair(-3, 1);
  result[1] = std::make_pair(-2, 8);
  result[2] = std::make_pair(-5, 3);
  result[3] = std::make_pair(5, 7);
  result[4] = std::make_pair(-7, -3);
  result[5] = std::make_pair(8, 3);
  result[6] = std::make_pair(1, 3);
  result[7] = std::make_pair(-1, 2);
  result[8] = std::make_pair(-4, 5);
  result[9] = std::make_pair(-8, 1);
  result[10] = std::make_pair(4, -2);
  result[11] = std::make_pair(-6, 1);
  result[12] = std::make_pair(2, -3);
  result[13] = std::make_pair(5, 4);
  result[14] = std::make_pair(-7, 1);
  result[15] = std::make_pair(1, 6);
  result[16] = std::make_pair(0, -6);
  result[17] = std::make_pair(9, 0);
  result[18] = std::make_pair(2, 1);
  result[19] = std::make_pair(2, 9);

  result = grahamScanParallel(result.begin(), result.end(), 4);

  std::vector<std::pair<double, double> > check(result.size());
  check[0] = std::make_pair(-8, 1);
  check[1] = std::make_pair(-7, -3);
  check[2] = std::make_pair(0, -6);
  check[3] = std::make_pair(9, 0);
  check[4] = std::make_pair(8, 3);
  check[5] = std::make_pair(5, 7);
  check[6] = std::make_pair(2, 9);
  check[7] = std::make_pair(-2, 8);

  ASSERT_EQ(result, check);
}

TEST(ConvexHull, getConvexHullGrahamTbb_3) {
  size_t size = 30;
  std::vector<std::pair<double, double> > points(size);

  points = generateRandomPoints(size);

  auto resultParallel = grahamScanParallel(points.begin(), points.end(), 4);
  auto resultSeq = grahamScan(points.begin(), points.end());

    for (size_t i = 0; i < resultParallel.size(); i++) {
        ASSERT_NEAR(resultParallel[i].first, resultSeq[i].first, 0.00001);
        ASSERT_NEAR(resultParallel[i].second, resultSeq[i].second, 0.00001);
    }
}

TEST(ConvexHull, getConvexHullGrahamTbb_4) {
  size_t size = 300;
  std::vector<std::pair<double, double> > points(size);

  points = generateRandomPoints(size);

  auto resultParallel = grahamScanParallel(points.begin(), points.end(), 4);
  auto resultSeq = grahamScan(points.begin(), points.end());

    for (size_t i = 0; i < resultParallel.size(); i++) {
        ASSERT_NEAR(resultParallel[i].first, resultSeq[i].first, 0.00001);
        ASSERT_NEAR(resultParallel[i].second, resultSeq[i].second, 0.00001);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
