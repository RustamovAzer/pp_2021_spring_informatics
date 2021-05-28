// Copyright 2021 Loganov Andrei
#include <time.h>
#include <omp.h>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "./qsort.h"
TEST(Q_SORT_OMP_TESTS, Test1) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec = {10.17, -0.45, 40, -1, 12, 34.27, 77, -0.4, 17};
    vec2 = {-1, -0.45, -0.4, 10.17, 12, 17, 34.27, 40, 77};
    QsortOMP(&vec);
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_OMP_TESTS, Test2) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    QsortOMP(&vec2);
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_OMP_TESTS, Test3) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec = getRandomVector(213);
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    QsortOMP(&vec2);
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_OMP_TESTS, Test4) {
    std::vector<double> vec;
    std::vector<double> vec2;
    vec = getRandomVector(564);
    vec2 = copyvec(vec);
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    QsortOMP(&vec2);
    ASSERT_EQ(vec2, vec);
}
TEST(Q_SORT_OMP_TESTS, Test5) {
    std::vector<double> vec;
    std::vector<double> vec2;
    // time_t t1, t2, t3, t4;
    vec = getRandomVector(600);
    vec2 = copyvec(vec);
    // t1 = clock();
    Hoarsort(&vec, 0, static_cast<int>(vec.size())-1);
    // t2 = clock();
    // double lin = t2 - t1;
    // std::cout << "lin" << lin / CLOCKS_PER_SEC << std::endl;
    // t3 = clock();
    QsortOMP(&vec2);
    // t4 = clock();
    // double par = t4 - t3;
    // std::cout << "par" << par / CLOCKS_PER_SEC << std::endl;
    // std::cout << "div" << lin / par << std::endl;
    ASSERT_EQ(vec2, vec);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
