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
TEST(Conjugate_gradient, MTX2x2) {
    std::vector<std::vector<double> > A(2);
    std::vector<double> vec(2);
    std::vector<double> res;

    vec[0] = 4;
    vec[1] = 8;

    A[0].push_back(2);
    A[0].push_back(6);
    A[1].push_back(6);
    A[1].push_back(6);

    res.push_back(1);
    res.push_back(0.333);
    std::vector<double> calc = calculateRes(A, vec);
    for (size_t i = 0; i < res.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
TEST(Conjugate_gradient, MTX3x3) {
    std::vector<double> vec(3);
    std::vector<std::vector<double> > mtx(3);
    vec[0] = 2;
    vec[1] = 1;
    vec[2] = 3;

    mtx[0].push_back(2);
    mtx[0].push_back(1);
    mtx[0].push_back(2);
    mtx[1].push_back(1);
    mtx[1].push_back(3);
    mtx[1].push_back(0);
    mtx[2].push_back(2);
    mtx[2].push_back(0);
    mtx[2].push_back(3);

    std::vector<double> res(3);
    res[0] = -1;
    res[1] = 0.666;
    res[2] = 1.666;
    std::vector<double> calc = calculateRes(mtx, vec);
    for (size_t i = 0; i < calc.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
TEST(Conjugate_gradient, MTX4x4) {
    std::vector<double> vec(4);
    std::vector<std::vector<double> > mtx(4);
    vec[0] = 3;
    vec[1] = 6;
    vec[2] = 2;
    vec[3] = 1;

    mtx[0].push_back(1);
    mtx[0].push_back(9);
    mtx[0].push_back(2);
    mtx[0].push_back(2);
    mtx[1].push_back(9);
    mtx[1].push_back(7);
    mtx[1].push_back(3);
    mtx[1].push_back(9);
    mtx[2].push_back(2);
    mtx[2].push_back(3);
    mtx[2].push_back(9);
    mtx[2].push_back(8);
    mtx[3].push_back(2);
    mtx[3].push_back(9);
    mtx[3].push_back(8);
    mtx[3].push_back(1);

    std::vector<double> res(4);
    res[0] = 0.1059;
    res[1] = 0.2881;
    res[2] = -0.2794;
    res[3] = 0.4297;
    std::vector<double> calc = calculateRes(mtx, vec);
    for (size_t i = 0; i < calc.size(); ++i) {
        ASSERT_NEAR(calc[i], res[i], 0.05);
    }
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
