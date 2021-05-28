// Copyright 2021 Yurin Stanislav
#include <gtest/gtest.h>
#include <vector>
#include "./strassen_algorithm.h"

#define EPS 0.00001

TEST(Strassen_algorithm, TEST1) {
    int rows_num = 2;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(-100, 100);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-100, 100);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST2) {
    int rows_num = 4;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(-31, 36);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-12, 89);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST3) {
    int rows_num = 8;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(4, 301);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-16, 0);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST4) {
    int rows_num = 16;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(-56, 90);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(0, 103);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST5) {
    int rows_num = 32;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(-150, 0);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-41, 300);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST6) {
    int rows_num = 64;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(-50, 81);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-2, 321);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

TEST(Strassen_algorithm, TEST7) {
    int rows_num = 128;

    std::vector<double> a(rows_num * rows_num);
    std::vector<double> b(rows_num * rows_num);


    for (int i = 0; i < static_cast<int>(a.size()); i++)
        a[i] = getRandomNumber(0, 11);

    for (int i = 0; i < static_cast<int>(b.size()); i++)
        b[i] = getRandomNumber(-15, 10);

    std::vector<double> c_strass(rows_num * rows_num);
    std::vector<double> c_default(rows_num * rows_num);
    std::vector<double>::iterator c_it = c_default.begin();

    c_strass = StrassenMulti(a, b);

    matrMulti(a, b, c_it, rows_num);

    for (int i = 0; i < static_cast<int>(c_strass.size()); i++) {
        ASSERT_LT(abs(c_strass[i] - c_default[i]), EPS);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
