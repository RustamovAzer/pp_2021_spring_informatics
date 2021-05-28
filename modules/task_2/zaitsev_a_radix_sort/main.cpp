// Copyright 2021 Zaitsev Andrey
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "./RadixSortOmp.h"

TEST(Test_Omp, Size_5_Without_Random) {
  int size = 5;
  double* vec = new double[size] {5.21, -8.652,
  26.32, 3.289, -14.842};
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec, tmp, size);
  result = check(vec, size);
  ASSERT_EQ(true, result);
  delete[] tmp;
  delete[] vec;
}

TEST(Test_Omp, Size_1000_Random_Pos_And_Neg) {
  int size = 1000;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec.data(), tmp, size);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
  delete[] tmp;
}

TEST(Test_Omp, Size_1000_Batcher_Seq_Border_20) {
  int size = 1000;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortSeq(vec.data(), size, 20);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Omp, Size_100_Batcher_Omp_1thr_Border_25) {
  int size = 100;
  int thr = 1;
  std::vector<double> vec = createVector(size, -100.0, 100.0);
  bool result = false;
  oddEvenMergeSortOmp(vec.data(), size, 25, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Omp, Size_100_Batcher_Omp_4thr_Border_25) {
  int size = 100;
  int thr = 4;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortOmp(vec.data(), size, 25, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Omp, Size_1000_Batcher_Omp_4thr_Border_60) {
  int size = 1000;
  int thr = 4;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortOmp(vec.data(), size, 60, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Omp, Test_Equality_Seq_Par_1000) {
  int size = 1000;
  int thr = 5;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  double t1 = omp_get_wtime();
  oddEvenMergeSortSeq(v1.data(), size, 50);
  double t2 = omp_get_wtime();
  std::cout << "Seq: " << t2 - t1 << std::endl;
  t1 = omp_get_wtime();
  oddEvenMergeSortOmp(v2.data(), size, 50, thr);
  t2 = omp_get_wtime();
  std::cout << "Omp: " << t2 - t1 << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Omp, Test_Equality_Seq_Par_2000) {
  int size = 2000;
  int thr = 5;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  double t1 = omp_get_wtime();
  oddEvenMergeSortSeq(v1.data(), size, 50);
  double t2 = omp_get_wtime();
  std::cout << "Seq: " << t2 - t1 << std::endl;
  t1 = omp_get_wtime();
  oddEvenMergeSortOmp(v2.data(), size, 50, thr);
  t2 = omp_get_wtime();
  std::cout << "Omp: " << t2 - t1 << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Omp, Test_Equality_Seq_Par_3000) {
  int size = 3000;
  int thr = 5;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  double t1 = omp_get_wtime();
  oddEvenMergeSortSeq(v1.data(), size, 50);
  double t2 = omp_get_wtime();
  std::cout << "Seq: " << t2 - t1 << std::endl;
  t1 = omp_get_wtime();
  oddEvenMergeSortOmp(v2.data(), size, 50, thr);
  t2 = omp_get_wtime();
  std::cout << "Omp: " << t2 - t1 << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Omp, Test_Equality_Seq_Par_4000) {
  int size = 4000;
  int thr = 5;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  double t1 = omp_get_wtime();
  oddEvenMergeSortSeq(v1.data(), size, 50);
  double t2 = omp_get_wtime();
  std::cout << "Seq: " << t2 - t1 << std::endl;
  t1 = omp_get_wtime();
  oddEvenMergeSortOmp(v2.data(), size, 50, thr);
  t2 = omp_get_wtime();
  std::cout << "Omp: " << t2 - t1 << std::endl;
  ASSERT_EQ(v1, v2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
