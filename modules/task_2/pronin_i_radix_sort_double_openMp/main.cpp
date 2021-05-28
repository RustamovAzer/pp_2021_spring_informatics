// Copyright 2021 Pronin Igor
#include <gtest/gtest.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include "./pronin_i_radix_sort_double_openMp.h"

TEST(Radix_sort_double, Positive_numbers7) {
    int size = 20000;
    int threads = 2;
    int left = 0;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    clock_t start = clock();
    SeqSort(vec, vec1, size, threads);
    clock_t end = clock();
    double result = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "SeqTime:" << result;
    double start1 = omp_get_wtime();
    ParSort(vec2, vec3, size, threads);
    double end1 = omp_get_wtime();
    double result1 = end1 - start1;
    std::cout << "ParTime:" << result1;
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec1[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, Negative_numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = -1;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, All_numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, 100numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, 1000numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
