// Copyright 2021 Bulychev Vladislav
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include <cmath>
#include <vector>
#include "./int_rec_tbb.h"

double f2(double x, double y) {
    return x + y + 1;
}

double f1(double x, double y) {
    return x * x * x - y * y;
}

TEST(Integrate_rectangle, DISABLED_Test_int_rec_1) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 17;
    a[1] = 21;
    b[1] = 53;

    tbb::tick_count t1 = tbb::tick_count::now();
    double ans1 = Calculation_Seq(a, b, 10000, f1);
    tbb::tick_count t2 = tbb::tick_count::now();
    std::cout << "1: " << ans1 << std::endl;
    std::cout << "Seq: " <<
        static_cast<double>((t2 - t1).seconds()) << std::endl;

    t1 = tbb::tick_count::now();
    double ans2 = Calculation_Tbb(a, b, 10000, f1);
    t2 = tbb::tick_count::now();
    std::cout << "2: " << ans2 << std::endl;
    std::cout << "Tbb: " <<
        static_cast<double>((t2 - t1).seconds()) << std::endl;

    ASSERT_NEAR(ans2, ans1, 5000);
}

TEST(Integrate_rectangle, Test_int_rec_2) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 14;
    a[1] = 3;
    b[1] = 21;

    ASSERT_NO_THROW(Calculation_Tbb(a, b, 5, f2));
}

TEST(Integrate_rectangle, Test_int_rec_3) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 14;
    a[1] = 3;
    b[1] = 21;

    ASSERT_NO_THROW(Calculation_Seq(a, b, 5, f2));
}

TEST(Integrate_rectangle, Test_int_rec_4) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 14;
    a[1] = 3;
    b[1] = 21;

    double ans1 = Calculation_Seq(a, b, 50, f1);
    double ans2 = Calculation_Tbb(a, b, 50, f1);

    ASSERT_NEAR(ans2, ans1, 10000);
}

TEST(Integrate_rectangle, Test_int_rec_5) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 17;
    a[1] = 21;
    b[1] = 53;

    double ans = 104704;

    double ans1 = Calculation_Seq(a, b, 100, f1);

    ASSERT_NEAR(ans1, ans, 50000);
}

TEST(Integrate_rectangle, Test_int_rec_6) {
    int s = 2;
    std::vector<double> a(s);
    std::vector<double> b(s);

    a[0] = 5;
    b[0] = 17;
    a[1] = 21;
    b[1] = 53;

    double ans = 104704;

    double ans1 = Calculation_Tbb(a, b, 100, f1);

    ASSERT_NEAR(ans1, ans, 50000);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
