// Copyright 2021 Chirkov Roman
#include <gtest/gtest.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "./jarvis_march.h"

TEST(jarvis_march_OMP, jarvis_bunch_of_points) {
  std::vector<Point> points = {Point(1, 1), Point(0, 0),    Point(10, 0),
                               Point(0, 6), Point(14, 200), Point(230, 15),
                               Point(4, 2)};
  std::vector<int> exp = {1, 2, 5, 4, 3, 1};
  std::vector<int> res = ParallelJarvis(points);

  ASSERT_EQ(exp, res);
}
TEST(jarvis_march_OMP, generation_wrong_size) {
  std::vector<Point> points;
  ASSERT_ANY_THROW(points = makePointsArray(0));
}

TEST(jarvis_march_OMP, jarvis_one_point) {
  std::vector<Point> points = {Point(123, 321)};
  std::vector<int> exp = {0, 0};
  std::vector<int> res = ParallelJarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march_OMP, jarvis_two_points) {
  std::vector<Point> points = {Point(123, 321), Point(1, 2)};
  std::vector<int> exp = {1, 0, 1};
  std::vector<int> res = ParallelJarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march_OMP, jarvis_two_bottom_points) {
  std::vector<Point> points = {Point(123, 321), Point(1, 2), Point(4, 2)};
  std::vector<int> exp = {1, 2, 0, 1};
  std::vector<int> res = ParallelJarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march_OMP, jarvis_three_points_on_the_same_line) {
  std::vector<Point> points = {Point(400, 400), Point(300, 400),
                               Point(100, 400)};
  std::vector<int> exp = {2, 0, 2};
  std::vector<int> res = ParallelJarvis(points);
  ASSERT_EQ(exp, res);
}

TEST(jarvis_march_OMP, generation_and_Jarvis) {
  std::vector<Point> points = makePointsArray(5000);
  // std::cout << "0" << std::endl;
  ASSERT_NO_THROW(ParallelJarvis(points));
}

TEST(jarvis_march_OMP, generationCircle_and_Jarvis) {
  std::vector<Point> points = makePointsArrayCircle(5000);
  // std::cout << "0" << std::endl;
  SequentialJarvis(points);
  ASSERT_NO_THROW(ParallelJarvis(points));
}

TEST(jarvis_march_OMP, seq_and_par_compare) {
  std::vector<Point> points = makePointsArray(100000);
  double start, stop;
  double seqTime, parTime;
  std::vector<int> seqHull, parHull;

  start = omp_get_wtime();
  seqHull = SequentialJarvis(points);
  stop = omp_get_wtime();
  seqTime = stop - start;
  std::cout << "seqTime = " << seqTime << std::endl;

  start = omp_get_wtime();
  parHull = ParallelJarvis(points);
  stop = omp_get_wtime();
  parTime = stop - start;
  std::cout << "parTime = " << parTime << std::endl;

  ASSERT_EQ(parHull, seqHull);
}

TEST(jarvis_march_OMP, circle_seq_and_par_compare) {
  std::vector<Point> points = makePointsArrayCircle(100000);
  double start, stop;
  double seqTime, parTime;
  std::vector<int> seqHull, parHull;

  start = omp_get_wtime();
  seqHull = SequentialJarvis(points);
  stop = omp_get_wtime();
  seqTime = stop - start;
  std::cout << "seqTime = " << seqTime << std::endl;

  start = omp_get_wtime();
  parHull = ParallelJarvis(points);
  stop = omp_get_wtime();
  parTime = stop - start;
  std::cout << "parTime = " << parTime << std::endl;

  ASSERT_EQ(parHull, seqHull);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
