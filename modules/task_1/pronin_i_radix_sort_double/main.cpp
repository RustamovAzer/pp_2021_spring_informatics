// Copyright 2021 Pronin Igor
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "./radix_sort_double.h"

TEST(Radix_sort_double, Positive_numbers) {
    int size = 10;
    int left = 0;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++)
        vec2[i] = vec[i];
    SortAll(vec, vec1, size);
    std::sort(vec2, vec2 + size);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec2[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
}
TEST(Radix_sort_double, Negative_numbers) {
    int size = 10;
    int left = -100;
    int right = -1;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++)
        vec2[i] = vec[i];
    SortAll(vec, vec1, size);
    std::sort(vec2, vec2 + size);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec2[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
}
TEST(Radix_sort_double, All_numbers) {
    int size = 10;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++)
        vec2[i] = vec[i];
    SortAll(vec, vec1, size);
    std::sort(vec2, vec2 + size);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec2[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
}
TEST(Radix_sort_double, 100numbers) {
    int size = 100;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++)
        vec2[i] = vec[i];
    SortAll(vec, vec1, size);
    std::sort(vec2, vec2 + size);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec2[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
}
TEST(Radix_sort_double, 1000numbers) {
    int size = 1000;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++)
        vec2[i] = vec[i];
    SortAll(vec, vec1, size);
    std::sort(vec2, vec2 + size);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec2[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
