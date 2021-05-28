// Copyright 2021 Kolesin Andrey
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "./radix.h"

TEST(RadixSort, 321) {
  std::vector<int> a = {3, 2, 1};
  radixSort(&a[0], a.size());
  ASSERT_EQ(a, std::vector<int>({1, 2, 3}));
}

TEST(RadixSort, 1) {
  std::vector<int> a = {1};
  radixSort(&a[0], a.size());
  ASSERT_EQ(a, std::vector<int>({1}));
}
TEST(RadixSort, negative) {
  std::vector<int> a = {-1, -2, -3};
  radixSort(&a[0], a.size());
  ASSERT_EQ(a, std::vector<int>({-3, -2, -1}));
}
TEST(RadixSort, mixed) {
  std::vector<int> a = {-1, -2, -3, 3, 1, 2, 0};
  radixSort(&a[0], a.size());
  ASSERT_EQ(a, std::vector<int>({-3, -2, -1, 0, 1, 2, 3}));
}

TEST(RadixSort, random) {
  std::vector<int> a = getRandomVector();
  // printVec(a);
  std::vector<int> b = a;
  radixSort(&a[0], a.size());
  std::sort(std::begin(b), std::end(b));
  // printVec(a);
  ASSERT_EQ(a, b);
}
