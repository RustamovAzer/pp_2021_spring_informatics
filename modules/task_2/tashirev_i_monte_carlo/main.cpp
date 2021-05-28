// Copyright 2021 Tashirev Ivan
#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include "../../../modules/task_2/tashirev_i_monte_carlo/monte_carlo.h"

double Integral_1(const std::vector<double>& x) {
    return x[0] * x[0];
}

double Integral_2(const std::vector<double>& x) {
    return 3 * x[0] * x[0] * x[0] + 2 * x[1] * x[1];
}

double Integral_3(const std::vector<double>& x) {
    return sin(x[0]) + 2 * x[1] + x[2] * x[2];
}

double Integral_4(const std::vector<double>& x) {
    return x[0] * x[0] + 2 * x[1] - cos(x[2]) + 2 * x[3] * x[3] * x[3] - 3 * x[4];
}


TEST(Monte_carlo_integral_test, test_result_of_integral) {
    double a = 0.0, b = 1.0;
    int num_th = 4;
    const int N = 2500000;
    omp_set_num_threads(num_th);

    double res_seq = seqMonteCarlo(Integral_1, { a }, { b }, N * num_th);
    double res_omp = ompMonteCarlo(Integral_1, { a }, { b }, N);

    ASSERT_NEAR(res_seq, res_omp, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_1) {
    std::vector<double> a = { 0.0, 2.5 };
    std::vector<double> b = { 1.534, 3.12 };
    int num_th = 4;
    const int N = 2500000;
    omp_set_num_threads(num_th);

    double res_seq = seqMonteCarlo(Integral_2, a, b, N * num_th);
    double res_omp = ompMonteCarlo(Integral_2, a, b, N);

    ASSERT_NEAR(res_seq, res_omp, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_2) {
    std::vector<double> a = { 0.0, 2.5, 1.234 };
    std::vector<double> b = { 1.534, 3.12, 1.555 };
    int num_th = 4;
    const int N = 2500000;
    omp_set_num_threads(num_th);

    double res_seq = seqMonteCarlo(Integral_3, a, b, N * num_th);
    double res_omp = ompMonteCarlo(Integral_3, a, b, N);

    ASSERT_NEAR(res_seq, res_omp, 0.5);
}

TEST(Monte_carlo_integral_test, test_result_of_integral_3) {
    std::vector<double>a = { 0.3, 1.32, 1.234 };
    std::vector<double>b = { 3.534, 3.5, 1.435 };
    int num_th = 4;
    const int N = 2500000;
    omp_set_num_threads(num_th);

    double res_seq = seqMonteCarlo(Integral_3, a, b, N * num_th);
    double res_omp = ompMonteCarlo(Integral_3, a, b, N);

    ASSERT_NEAR(res_seq, res_omp, 0.5);
}

TEST(Monte_carlo_integral_test, test_n_is_negative) {
    double a = 0.0, b = 3.0;
    ASSERT_ANY_THROW(ompMonteCarlo(Integral_1, { a }, { b }, -1000));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
