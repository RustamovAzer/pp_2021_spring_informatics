// Copyright 2021 Luckianchencko Ivan
#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "./obolochka.h"

TEST(obolocha, firts_matr) {
    mat a(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            a[i][j] = 0;
    }
    a[0][0] = 1;
    a[2][2] = 1;
    a[2][0] = 1;
    a[1][1] = 1;
    std::vector<point> res = Jarvis_alg(a);
    std::vector<int> cor1;
    std::vector<int> cor2;
    cor1.push_back(0);
    cor1.push_back(2);
    cor1.push_back(2);
    cor2.push_back(0);
    cor2.push_back(2);
    cor2.push_back(0);
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, cor1[i]);
        ASSERT_EQ(res[i].y, cor2[i]);
    }
}
TEST(obolocha, firts_matr1) {
    mat a(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            a[i][j] = 0;
    }
    a[0][0] = 1;
    a[2][2] = 1;
    std::vector<point> res = Jarvis_alg(a);
    std::vector<int> cor1;
    std::vector<int> cor2;
    cor1.push_back(0);
    cor1.push_back(2);
    cor2.push_back(0);
    cor2.push_back(2);
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, cor1[i]);
        ASSERT_EQ(res[i].y, cor2[i]);
    }
}
TEST(obolocha, firts_matr2) {
mat a(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            a[i][j] = 0;
    }
    a[0][1] = 1;
    a[1][0] = 1;
    a[1][2] = 1;
    a[1][1] = 1;
    std::vector<point> res = Jarvis_alg(a);
    std::vector<int> cor1;
    std::vector<int> cor2;
    cor1.push_back(1);
    cor1.push_back(1);
    cor1.push_back(0);
    cor2.push_back(0);
    cor2.push_back(2);
    cor2.push_back(1);
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, cor1[i]);
        ASSERT_EQ(res[i].y, cor2[i]);
    }
}
TEST(obolocha, firts_matr3) {
mat a(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            a[i][j] = 0;
    }
    a[0][1] = 1;
    a[0][0] = 1;
    a[0][2] = 1;
    std::vector<point> res = Jarvis_alg(a);
    std::vector<int> cor1;
    std::vector<int> cor2;
    cor1.push_back(0);
    cor1.push_back(0);
    cor2.push_back(0);
    cor2.push_back(2);
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, cor1[i]);
        ASSERT_EQ(res[i].y, cor2[i]);
    }
}
TEST(obolocha, firts_matr4) {
    mat a(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            a[i][j] = 0;
    }
    a[0][1] = 1;
    a[1][0] = 1;
    a[1][2] = 1;
    a[1][1] = 1;
    std::vector<point> res = Jarvis_alg(a);
    std::vector<int> cor1;
    std::vector<int> cor2;
    cor1.push_back(1);
    cor1.push_back(1);
    cor1.push_back(0);
    cor2.push_back(0);
    cor2.push_back(2);
    cor2.push_back(1);
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, cor1[i]);
        ASSERT_EQ(res[i].y, cor2[i]);
    }
}


