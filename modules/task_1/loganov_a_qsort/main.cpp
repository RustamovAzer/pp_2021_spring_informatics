// Copyright 2021 Loganov Andrei
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "./qsort.h"
TEST(Q_SORT_TESTS, Test1) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec = {10.17, -0.45, 40, -1, 12, 34.27, 77, -0.4, 17};
    vec2 = {-1, -0.45, -0.4, 10.17, 12, 17, 34.27, 40, 77};
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_TESTS, Test2) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    std::sort(vec2.begin(), vec2.end(),
    [](double a, double b) {return a < b; });
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_TESTS, Test3) {
     std::vector<double> vec;
    std::vector<double> vec2;
    vec = getRandomVector(5000);
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    std::sort(vec2.begin(), vec2.end(),
    [](double a, double b) {return a < b; });
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_TESTS, Test4) {
     std::vector<double> vec;
    std::vector<double> vec2;
    vec = getRandomVector(1);
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    std::sort(vec2.begin(), vec2.end(),
    [](double a, double b) {return a < b; });
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_TESTS, Test5) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec = getRandomVector(50000);
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    std::sort(vec2.begin(), vec2.end(),
    [](double a, double b) {return a < b; });
    ASSERT_EQ(vec2, vec);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
