// Copyright 2021 Volkova Anastasia
#include <gtest/gtest.h>
#include "./rectangle_method_std.h"

TEST(Integral, Test1_10) {
    double integral = integralFunction(function1, 0, 17, -1, 2, 4, 10, 10, 10, 10);
    double parallel_integral = ParallelIntegralFunction(function1, 0, 17, -1, 2, 4, 10, 10, 10, 10);
    double value = 9103.5;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test1_20) {
    double integral = integralFunction(function1, 0, 17, -1, 2, 4, 10, 20, 20, 20);
    double parallel_integral = ParallelIntegralFunction(function1, 0, 17, -1, 2, 4, 10, 20, 20, 20);
    double value = 9103.5;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test2_50) {
    double integral = integralFunction(function1, -1, 1.9, -1.3, 2, 2, 6.5, 50, 50, 50);
    double parallel_integral = ParallelIntegralFunction(function1, -1, 1.9, -1.3, 2, 2, 6.5, 50, 50, 50);
    double value = 28.826634375;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test3_90) {
    double integral = integralFunction(function3, 0, 1.8, -1.2, 1.1, -1, 2.9, 90, 90, 90);
    double parallel_integral = ParallelIntegralFunction(function3, 0, 1.8, -1.2, 1.1, -1, 2.9, 90, 90, 90);
    double value = 67.49028;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test4_100) {
    double integral = integralFunction(function3, 0.5, 1.7, -1, 2, -4, 1.9, 100, 100, 100);
    double parallel_integral = ParallelIntegralFunction(function3, 0.5, 1.7, -1, 2, -4, 1.9, 100, 100, 100);
    double value = -144.963;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test5_20) {
    double integral = integralFunction(function1, 0.1, 1, -1, 2, 4.2, 6, 20, 20, 20);
    double parallel_integral = ParallelIntegralFunction(function1, 0.1, 1, -1, 2, 4.2, 6, 20, 20, 20);
    double value = 6.81615;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}

TEST(Integral, Test6_75) {
    double integral = integralFunction(function1, 0.1, 1, -1.1, 2, -1, 5, 75, 75, 75);
    double parallel_integral = ParallelIntegralFunction(function1, 0.1, 1, -1.1, 2, -1, 5, 75, 75, 75);
    double value = 8.2863;
    ASSERT_LE(std::abs(parallel_integral - value), 1e-3);
    std::cout << "value: " << value << std::endl;
    std::cout << "integral: " << integral << std::endl;
    std::cout << "parallel integral: " << parallel_integral << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
