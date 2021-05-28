// Copyright 2021 Gushchin Artem
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "./simpsonMethod.h"

TEST(SimpsonsMethodIntegrating, simplestConstFunction) {
    auto func = [](const std::vector<double>& arguments) { return 8; };
    std::vector<std::pair<double, double>> segments = { {-3, 1} };
    ASSERT_NEAR(simpsonsMethod(func, segments, 10), 32, 0.00001);
}

TEST(SimpsonsMethodIntegrating, oneDimentionalFunction) {
    auto func = [](const std::vector<double>& arguments) {
        return pow(arguments[0], 3) + cos(arguments[0]);
    };
    std::vector<std::pair<double, double>> segments = { {-5, 4} };
    ASSERT_NEAR(simpsonsMethod(func, segments, 100), -93.96572, 0.00001);
}

TEST(SimpsonsMethodIntegrating, twoDimentionalFunction) {
    auto func = [](const std::vector<double>& arguments) {
        return pow(arguments[0], 2) + pow(arguments[1], 2);
    };
    std::vector<std::pair<double, double>> segments = { {-1, 3}, {7, 9} };
    ASSERT_NEAR(simpsonsMethod(func, segments, 100), 533.33333, 0.00001);
}

TEST(SimpsonsMethodIntegrating, threeDimentionalFunction) {
    auto func = [](const std::vector<double>& arguments) {
        return arguments[0] + arguments[1] + cos(arguments[2]);
    };
    std::vector<std::pair<double, double>> segments = { {0, 5}, {-2, 3},
                                                        { -1, 3 } };
    ASSERT_NEAR(simpsonsMethod(func, segments, 100), 324.56477, 0.00001);
}

TEST(SimpsonsMethodIntegrating, throwsWhenSegmentsAreEmpty) {
    auto func = [](const std::vector<double>& arguments) {
        return arguments[0] + arguments[1] + cos(arguments[2]);
    };
    std::vector<std::pair<double, double>> segments;
    ASSERT_ANY_THROW(simpsonsMethod(func, segments, 100));
}

TEST(SimpsonsMethodIntegrating, throwsWhenStepsCountIsZero) {
    auto func = [](const std::vector<double>& arguments) {
        return arguments[0] + arguments[1] + cos(arguments[2]);
    };
    std::vector<std::pair<double, double>> segments = { { 0, 5 },
                                                        { -2, 3 },
                                                        { -1, 3 } };
    ASSERT_ANY_THROW(simpsonsMethod(func, segments, 0));
}
