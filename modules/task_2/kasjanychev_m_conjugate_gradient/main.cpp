// Copyright 2021 Kasjanychev Mikhail
#include <gtest/gtest.h>
#include "./conjugate_gradient.h"

TEST(Conjugate_gradient, ScalarProduct) {
    std::vector<double> v1;
    std::vector<double> v2;

    v1.push_back(3);
    v1.push_back(1);
    v2.push_back(2);
    v2.push_back(5);

    double res = 11;
    ASSERT_EQ(res, scalarProduct(v1, v2));
}
TEST(Conjugate_gradient, MultiMtxVec) {
    std::vector<std::vector<double> > mtx(3);
    std::vector<double> v;

    mtx[0].push_back(3);
    mtx[0].push_back(2);
    mtx[0].push_back(5);
    mtx[1].push_back(0);
    mtx[1].push_back(1);
    mtx[1].push_back(4);
    mtx[2].push_back(3);
    mtx[2].push_back(6);
    mtx[2].push_back(1);

    v.push_back(2);
    v.push_back(4);
    v.push_back(1);

    std::vector<double> res;
    res.push_back(19);
    res.push_back(8);
    res.push_back(31);
    std::vector<double> calc = multiMtxVec(mtx, v);
    for (size_t i = 0; i < res.size(); ++i) {
        ASSERT_EQ(res[i], calc[i]);
    }
}
TEST(Conjugate_gradient, MTX10x10) {
    int n = 10;
    int num_of_threads = 2;
    std::vector<std::vector<double> > A = getRandomMatrix(n);
    std::vector<double> vec = getRandomVector(n);
    std::vector<double> res;

    res = calculateStandardRes(A, vec, num_of_threads);
    std::vector<double> calc = calculateRes(A, vec, num_of_threads);
    for (size_t i = 0; i < res.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
TEST(Conjugate_gradient, MTX50x50) {
    int n = 50;
    int num_of_threads = 3;
    std::vector<std::vector<double> > A = getRandomMatrix(n);
    std::vector<double> vec = getRandomVector(n);
    std::vector<double> res;

    res = calculateStandardRes(A, vec, num_of_threads);
    std::vector<double> calc = calculateRes(A, vec, num_of_threads);
    for (size_t i = 0; i < res.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
TEST(Conjugate_gradient, MTX100x100) {
    int n = 100;
    int num_of_threads = 5;
    std::vector<std::vector<double> > A = getRandomMatrix(n);
    std::vector<double> vec = getRandomVector(n);
    std::vector<double> res;

    clock_t start = clock();
    std::vector<double> calcSeq = calculateRes(A, vec, 1);
    clock_t end = clock();
    double result = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "SeqTime:" << result;

    double start1 = omp_get_wtime();
    std::vector<double> calc = calculateRes(A, vec, num_of_threads);
    double end1 = omp_get_wtime();
    double result1 = end1 - start1;
    std::cout << "\nParTime:" << result1 << "\n";

    res = calculateStandardRes(A, vec, num_of_threads);
    for (size_t i = 0; i < res.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
