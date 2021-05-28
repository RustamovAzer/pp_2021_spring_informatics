// Copyright 2021 Zaitsev Andrey
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./RadixSortSeq.h"

TEST(Test_Seq, Size_5_Without_Random) {
  int size = 5;
  double* vec = new double[size] {4.33, -6.541,
    16.84, 2.662, -12.544};
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Size_1000_Random_Positive) {
  int size = 1000;
  double* vec = new double[size];
  vec = createVector(vec, size, 100.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Size_1000_Random_Negative) {
  int size = 1000;
  double* vec = new double[size];
  vec = createVector(vec, size, -800.0, 0.0);
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Size_1000_Random_Pos_And_Neg) {
  int size = 1000;
  double* vec = new double[size];
  vec = createVector(vec, size, -500.0, 500.0);
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Exact_Answer) {
  int size = 7;
  double* vec = new double[size] {5.32, 8.25, 3.463, 43.733,
    120.632, 75.86, 53.41};
  double* answer = new double[size] {3.463, 5.32, 8.25,
    43.733, 53.41, 75.86, 120.632};
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
  delete[] answer;
}

TEST(Test_Seq, Size_100_Batcher) {
  int size = 100;
  double* vec = new double[size];
  vec = createVector(vec, size, 0.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  oddEvenMergeSort(vec, tmp, size, 6);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Size_1000_Batcher_Border_20) {
  int size = 1000;
  double* vec = new double[size];
  vec = createVector(vec, size, -1000.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  oddEvenMergeSort(vec, tmp, size, 20);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Seq, Size_1000_Batcher_Border_1000) {
  int size = 1000;
  double* vec = new double[size];
  vec = createVector(vec, size, -1000.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  oddEvenMergeSort(vec, tmp, size, 1000);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
