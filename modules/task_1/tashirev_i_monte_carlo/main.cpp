// Copyright 2021 Tashirev Ivan
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "./monte_carlo.h"

double seqIntegral_1(std::vector<double> x) {
    return x[0] * x[0];
}

double seqIntegral_2(std::vector<double> x) {
    return 3 * x[0] * x[0] * x[0] + 2 * x[1] * x[1];
}

double seqIntegral_3(std::vector<double> x) {
    return sin(x[0]) + 2 * x[1] + x[2] * x[2];
}

double seqIntegral_4(std::vector<double> x) {
    return x[0] * x[0] + 2 * x[1] - cos(x[2]) + 2 * x[3] * x[3] * x[3] - 3 * x[4];
}


TEST(Monte_carlo_integral_test, test_result_of_integral) {
    double a = 0.0, b = 1.0;
    double res = seqMonteCarlo(seqIntegral_1, { a }, { b }, 1000000);
    ASSERT_NEAR(res, 0.333, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_1) {
    std::vector<double>a = { 0.0, 2.5 };
    std::vector<double>b = { 1.534, 3.12 };
    double res = seqMonteCarlo(seqIntegral_2, a, b, 1000000);
    ASSERT_NEAR(res, 17.660, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_2) {
    std::vector<double>a = { 0.0, 2.5, 1.234 };
    std::vector<double>b = { 1.534, 3.12, 1.555 };
    double res = seqMonteCarlo(seqIntegral_3, a, b, 1000000);
    ASSERT_NEAR(res, 2.503, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_3) {
    std::vector<double>a = { 0.3, 3.5, 1.234 };
    std::vector<double>b = { 3.534, 1.32, 1.435 };
    double res = seqMonteCarlo(seqIntegral_3, a, b, 1000000);
    ASSERT_NEAR(res, -10.181, 0.5);
}

TEST(Monte_carlo_integral_test, test_n_is_negative) {
    double a = 0.0, b = 3.0;
    ASSERT_ANY_THROW(seqMonteCarlo(seqIntegral_1, { a }, { b }, -1000));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
