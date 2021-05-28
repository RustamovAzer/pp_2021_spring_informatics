// Copyright 2020 Bakaeva Maria
#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include "./convex_hull_graham.h"

TEST(ConvexHull, rotateTest) {
    std::pair<double, double> A(0, 4);
    std::pair<double, double> B(3, 2);
    std::pair<double, double> C(4.5, 5);

    ASSERT_DOUBLE_EQ(rotate(A, B, C), 12);
}

TEST(ConvexHull, getLeftmostPointTest) {
  std::vector<std::pair<double, double> > setOfPoints(5);

  setOfPoints[0] = std::make_pair(-1, -9);
  setOfPoints[1] = std::make_pair(0, 0);
  setOfPoints[2] = std::make_pair(9, 3);
  setOfPoints[3] = std::make_pair(4, 12);
  setOfPoints[4] = std::make_pair(-0.5, -4);

  std::pair<double, double> leftmostPoint = setOfPoints[0];
  std::swap(setOfPoints[0], setOfPoints[time(0) % setOfPoints.size()]);
  size_t leftmost = getLeftmostPoint(setOfPoints);

  ASSERT_EQ(setOfPoints[leftmost], leftmostPoint);
}

TEST(ConvexHull, sortPoints) {
  std::vector<std::pair<double, double> > setOfPoints(5);
  setOfPoints[0] = std::make_pair(-1, -9);
  setOfPoints[1] = std::make_pair(0, 0);
  setOfPoints[2] = std::make_pair(9, 3);
  setOfPoints[3] = std::make_pair(4, 12);
  setOfPoints[4] = std::make_pair(-0.5, -4);

  std::vector<std::pair<double, double> > resultPoints(5);
  resultPoints[0] = std::make_pair(-1, -9);
  resultPoints[1] = std::make_pair(9, 3);
  resultPoints[2] = std::make_pair(4, 12);
  resultPoints[3] = std::make_pair(0, 0);
  resultPoints[4] = std::make_pair(-0.5, -4);

  setOfPoints = sortPoints(setOfPoints);

  ASSERT_EQ(setOfPoints, resultPoints);
}

TEST(ConvexHull, getConvexHullGraham_1) {
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

  result = grahamScan(result);

  std::vector<std::pair<double, double> > check(result.size());
  check[0] = std::make_pair(-4, 0);
  check[1] = std::make_pair(1, -3);
  check[2] = std::make_pair(4, -2);
  check[3] = std::make_pair(5, 5);
  check[4] = std::make_pair(1, 7);
  check[5] = std::make_pair(-2, 6);

  ASSERT_EQ(result, check);
}

TEST(ConvexHull, getConvexHullGraham_2) {
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

  result = grahamScan(result);

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
