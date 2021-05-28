// Copyright 2021 Chirkov Roman
#include <gtest/gtest.h>
#include <vector>
#include "./jarvis_march.h"

TEST(jarvis_march, generation_wrong_size) {
  std::vector<Point> points;
  ASSERT_ANY_THROW(points = makePointsArray(0));
}

TEST(jarvis_march, jarvis_one_point) {
  std::vector<Point> points = {Point(123, 321)};
  std::vector<int> exp = {0, 0};
  std::vector<int> res = Jarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march, jarvis_two_points) {
  std::vector<Point> points = {Point(123, 321), Point(1, 2)};
  std::vector<int> exp = {1, 0, 1};
  std::vector<int> res = Jarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march, jarvis_two_bottom_points) {
  std::vector<Point> points = {Point(123, 321), Point(1, 2), Point(4, 2)};
  std::vector<int> exp = {1, 2, 0, 1};
  std::vector<int> res = Jarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march, generation_and_Jarvis) {
  std::vector<Point> points = makePointsArray(10);
  ASSERT_NO_THROW(Jarvis(points));
}

TEST(jarvis_march, jarvis_bunch_of_points) {
  std::vector<Point> points = {Point(1, 1), Point(0, 0),    Point(10, 0),
                               Point(0, 6), Point(14, 200), Point(230, 15),
                               Point(4, 2)};
  std::vector<int> exp = {1, 2, 5, 4, 3, 1};
  std::vector<int> res = Jarvis(points);

  ASSERT_EQ(exp, res);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
