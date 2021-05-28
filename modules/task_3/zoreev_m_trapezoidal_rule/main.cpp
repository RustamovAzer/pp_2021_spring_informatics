// Copyright 2021 Zoreev Mikhail

#include <gtest/gtest.h>
#include <tbb/tick_count.h>

#include <cmath>
#include <functional>
#include <iostream>

#include "../../../modules/task_3/zoreev_m_trapezoidal_rule/trapezoidal_rule.h"

TEST(TrapezoidalRule, Manual_Speed_Test) {
    std::vector<double> from = {0, 0};
    std::vector<double> to = {1, 1};
    std::function<double(std::vector<double>)> f = [](std::vector<double> args) {
        return std::sin(args[0]) * args[1];
    };
    tbb::tick_count  parallel_begin = tbb::tick_count::now();
    double parallel_result = integrate_parallel(f, from, to, 200);
    tbb::tick_count  parallel_end = tbb::tick_count::now();
    std::cout << "Parallel time:  " << (parallel_end - parallel_begin).seconds() << std::endl;
    tbb::tick_count seqential_begin = tbb::tick_count::now();
    double sequential_result = integrate(f, from, to, 200);
    tbb::tick_count seqential_end = tbb::tick_count::now();
    std::cout << "Seqential time: " << (seqential_end - seqential_begin).seconds() << std::endl;
    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
}

TEST(TrapezoidalRule, Can_Integrate_1Dimensional) {
    std::vector<double> from = {1};
    std::vector<double> to = {2};
    double result = integrate_parallel([](std::vector<double> args) {
        return std::log(args[0]);
        },
        from, to, 200);
    ASSERT_NEAR(0.3862943611198906, result, 1e-2);
}

TEST(TrapezoidalRule, Can_Integrate_2Dimensional) {
    std::vector<double> from = {0, 0};
    std::vector<double> to = {1, 1};
    double result = integrate_parallel([](std::vector<double> args) {
        return std::sin(args[0]) * args[1];
        },
        from, to, 200);
    ASSERT_NEAR(0.22984884706593, result, 1e-2);
}

TEST(TrapezoidalRule, Can_Integrate_3Dimensional) {
    std::vector<double> from = {0, 0, 0};
    std::vector<double> to = {1, 1, 1};
    double result = integrate_parallel([](std::vector<double> args) {
        return args[0] + args[1] + args[2];
        },
        from, to, 75);
    ASSERT_NEAR(1.5, result, 5e-2);
}

TEST(TrapezoidalRule, Cant_Integrate_Zero_Steps) {
    std::vector<double> from = {0};
    std::vector<double> to = {1};
    ASSERT_ANY_THROW(integrate_parallel([](std::vector<double> args) {
        return 1;
        },
        from, to, 0));
}

TEST(TrapezoidalRule, Cant_Integrate_With_Wrong_Range_Vectors) {
    std::vector<double> from = {0};
    std::vector<double> to = {1, 2};
    ASSERT_ANY_THROW(integrate_parallel([](std::vector<double> args) {
        return 1;
        },
        from, to, 200));
}

TEST(TrapezoidalRule, Cant_Integrate_With_Empty_Range_Vectors) {
    std::vector<double> from = {};
    std::vector<double> to = {};
    ASSERT_ANY_THROW(integrate_parallel([](std::vector<double> args) {
        return 1;
        },
        from, to, 200));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
