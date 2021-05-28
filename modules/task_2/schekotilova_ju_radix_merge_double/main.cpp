// Copyright 2021 Schekotilova Julia
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/schekotilova_ju_radix_merge_double/radix_merge_double.h"

TEST(TEST_OMP, TEST_SEQ_SIZE_5) {
  std::vector<double> vec = std::vector<double>({ 1.65, 5.5582, 29.9, 9.155, 52.864 });
  std::vector<double> res_v = radix_sort(vec);
  bool res = false;
  res = checker(res_v, 5);
  ASSERT_EQ(res, true);
}

TEST(TEST_OMP, TEST_SIZE_5) {
  std::vector<double> vec = std::vector<double>({ 26.048, -1.52, 20.22, 8.46, 92.942 });
  std::vector<double> res_v = radix_sort_omp(vec);
  bool res = false;
  res = checker(res_v, 5);
  ASSERT_EQ(res, true);
}

TEST(TEST_OMP, TEST_SIZE_3_3) {
  int size = 3;
  std::vector<double> vec = std::vector<double>({8.53, 73.4, 63.26});
  std::vector<double> vec_ = std::vector<double>({8.23, 56.2, -0.25});
  std::vector<double> res_v = radix_sort_omp(vec), res_v_ = radix_sort_omp(vec_);
  std::vector<double> result = merge(res_v, res_v_);
  bool res = false;
  res = checker(result, 2 * size);
  ASSERT_EQ(res, true);
}

TEST(TEST_OMP, TEST_RANDOM) {
  int size = 4;
  std::vector<double> vec = generate(size), vec_ = generate(size);
  std::vector<double> res_v = radix_sort_omp(vec), res_v_ = radix_sort_omp(vec_);
  std::vector<double> result = merge(res_v, res_v_);
  bool res = false;
  res = checker(result, size + size);
  ASSERT_EQ(res, true);
}

TEST(TEST_OMP, TEST_CHECK_TIME) {
  bool res = false;
  int size = 100000;
  std::vector<double> vec_ = generate(size);
  std::vector<double> vec = vec_;
  double time_start = omp_get_wtime();
  std::vector<double> result = radix_sort_omp(vec);
  double time_end = omp_get_wtime();
  res = checker(result, size);
  double time_st = omp_get_wtime();
  std::vector<double> seq = radix_sort(vec_);
  double time_e = omp_get_wtime();
  std::cout << "seq time: " << time_e - time_st << std::endl;
  std::cout << "omp time: " << time_end - time_start << std::endl;
  ASSERT_EQ(true, res);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
