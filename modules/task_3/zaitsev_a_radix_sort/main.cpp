// Copyright 2021 Zaitsev Andrey
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include <iostream>
#include <vector>
#include "./RadixSortTbb.h"

TEST(Test_Tbb, Size_5_Without_Random) {
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

TEST(Test_Tbb, Size_1000_Random_Pos_And_Neg) {
  int size = 1000;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  double* tmp = new double[size];
  bool result = false;
  radixSort(vec.data(), tmp, size);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
  delete[] tmp;
}

TEST(Test_Tbb, Size_1000_Batcher_Seq_Border_20) {
  int size = 1000;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortSeq(vec.data(), size, 20);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Tbb, Size_100_Batcher_Tbb_1thr_Border_25) {
  int size = 100;
  int thr = 1;
  std::vector<double> vec = createVector(size, -100.0, 100.0);
  bool result = false;
  oddEvenMergeSortTbb(vec.data(), size, 25, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Tbb, Size_100_Batcher_Tbb_4thr_Border_25) {
  int size = 100;
  int thr = 4;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortTbb(vec.data(), size, 25, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Tbb, Size_1000_Batcher_Tbb_4thr_Border_60) {
  int size = 1000;
  int thr = 4;
  std::vector<double> vec = createVector(size, -1000.0, 1000.0);
  bool result = false;
  oddEvenMergeSortTbb(vec.data(), size, 60, thr);
  result = check(vec.data(), size);
  ASSERT_EQ(true, result);
}

TEST(Test_Tbb, Test_Equality_Seq_Par_10000) {
  int size = 10000;
  int thr = 6;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  tbb::tick_count t1 = tbb::tick_count::now();
  oddEvenMergeSortSeq(v1.data(), size, 100);
  tbb::tick_count t2 = tbb::tick_count::now();
  std::cout << "Seq: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  t1 = tbb::tick_count::now();
  oddEvenMergeSortTbb(v2.data(), size, 100, thr);
  t2 = tbb::tick_count::now();
  std::cout << "Tbb: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Tbb, Test_Equality_Seq_Par_20000) {
  int size = 20000;
  int thr = 6;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  tbb::tick_count t1 = tbb::tick_count::now();
  oddEvenMergeSortSeq(v1.data(), size, 100);
  tbb::tick_count t2 = tbb::tick_count::now();
  std::cout << "Seq: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  t1 = tbb::tick_count::now();
  oddEvenMergeSortTbb(v2.data(), size, 100, thr);
  t2 = tbb::tick_count::now();
  std::cout << "Tbb: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Tbb, Test_Equality_Seq_Par_30000) {
  int size = 30000;
  int thr = 6;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  tbb::tick_count t1 = tbb::tick_count::now();
  oddEvenMergeSortSeq(v1.data(), size, 100);
  tbb::tick_count t2 = tbb::tick_count::now();
  std::cout << "Seq: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  t1 = tbb::tick_count::now();
  oddEvenMergeSortTbb(v2.data(), size, 100, thr);
  t2 = tbb::tick_count::now();
  std::cout << "Tbb: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Tbb, Test_Equality_Seq_Par_50000) {
  int size = 50000;
  int thr = 6;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  tbb::tick_count t1 = tbb::tick_count::now();
  oddEvenMergeSortSeq(v1.data(), size, 100);
  tbb::tick_count t2 = tbb::tick_count::now();
  std::cout << "Seq: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  t1 = tbb::tick_count::now();
  oddEvenMergeSortTbb(v2.data(), size, 100, thr);
  t2 = tbb::tick_count::now();
  std::cout << "Tbb: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  ASSERT_EQ(v1, v2);
}

TEST(Test_Tbb, Test_Equality_Seq_Par_100000) {
  int size = 100000;
  int thr = 6;
  std::vector<double> v1 = createVector(size, -1000.0, 1000.0);
  std::vector<double> v2 = v1;
  tbb::tick_count t1 = tbb::tick_count::now();
  oddEvenMergeSortSeq(v1.data(), size, 100);
  tbb::tick_count t2 = tbb::tick_count::now();
  std::cout << "Seq: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  t1 = tbb::tick_count::now();
  oddEvenMergeSortTbb(v2.data(), size, 100, thr);
  t2 = tbb::tick_count::now();
  std::cout << "Tbb: " <<
    static_cast<double>((t2 - t1).seconds()) << std::endl;
  ASSERT_EQ(v1, v2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
