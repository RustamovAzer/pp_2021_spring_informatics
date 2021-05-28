// Copyright 2021 Bulychev Vladislav
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "./int_rec_omp.h"

double f(std::vector<double> t) {
    double x = t[0];
    double y = t[1];
    return x * x - y * y;
}

double f5(std::vector<double> t) {
    double x = t[0];
    double y = t[1];
    return x * x + y * y;
}

double f3(std::vector<double> t) {
    double x = t[0];
    double y = t[1];
    return x + y + 1;
}

double f2(std::vector<double> t) {
    double x = t[0];
    return x * x - x;
}

double f1(std::vector<double> t) {
    double x = t[0];
    double y = t[1];
    double z = t[2];
    return (z - (5 * x + 2 * y));
}

double f4(std::vector<double> t) {
    double x = t[0];
    double y = t[1];
    return x * 10 - x * y + cos(y);
}

TEST(Integrate_rectangle, Test_int_rec_1) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 0;
    b[0] = 3;
    a[1] = 0;
    b[1] = 3;

    double t1 = omp_get_wtime();
    double ans1 = Calculation_Seq(a, b, 100, f5);
    double t2 = omp_get_wtime();
    std::cout << "1: " << ans1 << std::endl;
    std::cout << "Seq: " << t2 - t1 << std::endl;
    t1 = omp_get_wtime();
    double ans2 = Calculation_Omp(a, b, 100, f5);
    t2 = omp_get_wtime();
    std::cout << "2: " << ans2 << std::endl;
    std::cout << "Omp: " << t2 - t1 << std::endl;

    ASSERT_NEAR(ans1, ans2, 100);
}

TEST(Integrate_rectangle, Test_int_rec_2) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 0;
    b[0] = 5;
    a[1] = 0;
    b[1] = 3;

    double ans = 79;

    double ans1 = Calculation_Omp(a, b, 10, f);

    ASSERT_NEAR(ans, ans1, 100);
}

TEST(Integrate_rectangle, Test_int_rec_3) {
    int s = 1;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 25;

    double ans = 4867;

    double ans1 = Calculation_Omp(a, b, 100, f2);

    ASSERT_NEAR(ans, ans1, 1000);
}

TEST(Integrate_rectangle, Test_int_rec_4) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 0;
    b[0] = 1;
    a[1] = 2;
    b[1] = 4;

    double ans = 9;

    double ans1 = Calculation_Omp(a, b, 10, f3);

    ASSERT_NEAR(ans, ans1, 10);
}

TEST(Integrate_rectangle, Test_int_rec_5) {
    int s = 3;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 4;
    b[0] = 10;
    a[1] = 7;
    b[1] = 33;
    a[2] = -1;
    b[2] = 5;

    ASSERT_NO_THROW(Calculation_Omp(a, b, 5, f1));
}

TEST(Integrate_rectangle, Test_int_rec_6) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 4;
    b[0] = 10;
    a[1] = 7;
    b[1] = 33;

    ASSERT_NO_THROW(Calculation_Omp(a, b, 5, f4));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
