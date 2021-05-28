// Copyright 2021 Volkova Anastasia
#include <tbb/tbb.h>
#include <tbb/tick_count.h>
#include <iostream>
#include "../../../modules/task_3/volkova_a_rectangle_method/rectangle_method_tbb.h"

double function1(double x, double y, double z) {
    return x*y*z;
}
double function2(double x, double y, double z) {
    return x * x * x * y + 2 * z * z * y + 10 * x * y * z;
}
double function3(double x, double y, double z) {
    return (5 * x + 2 * y) * z;
}
double function4(double x, double y, double z) {
    return x * x + x * y * z - 10*z * z;
}

double integralFunction(double (*f)(double, double, double),
                        double ax, double bx,
                        double ay, double by,
                        double az, double bz,
                        int n, int m, int k) {
    tbb::tick_count start_calc = tbb::tick_count::now();
    const double hx = (bx - ax) / static_cast<double>(n);
    const double hy = (by - ay) / static_cast<double>(m);
    const double hz = (bz - az) / static_cast<double>(k);

    double ans = 0;
    double X, Y, Z;
    for (int i = 0; i < n; ++i) {
        X = ax + static_cast<double>(i) * hx + 0.5 * hx;
        for (int j = 0; j < m; ++j) {
            Y = ay + static_cast<double>(j) * hy + 0.5 * hy;
            for (int q = 0; q < k;  ++q) {
                Z = az + static_cast<double>(q) * hz + 0.5 * hz;
                ans += f(X, Y, Z);
            }
        }
    }
    ans *= hx * hy * hz;
    tbb::tick_count end_calc = tbb::tick_count::now();
    std::cout << "Runtime: " << static_cast<double>((end_calc - start_calc).seconds()) << std::endl;
    return ans;
}

double ParallelIntegralFunction(double (*f)(double, double, double),
                        double ax, double bx,
                        double ay, double by,
                        double az, double bz,
                        int n, int m, int k) {
    tbb::tick_count start = tbb::tick_count::now();
    const double hx = (bx - ax) / static_cast<double>(n);
    const double hy = (by - ay) / static_cast<double>(m);
    const double hz = (bz - az) / static_cast<double>(k);

    auto integral = tbb::parallel_reduce(
        tbb::blocked_range2d<int>(0, n, 0, m), 0.0,
        [&](tbb::blocked_range2d<int> r, double ans) -> double {
            for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                double X = ax + (i) * hx + 0.5 * hx;
                for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                    double Y = ay + (j) * hy + 0.5 * hy;
                    double sum = 0;
                    for (int q = 0; q < k;  ++q) {
                        double Z = az + (q) * hz + 0.5 * hz;
                        sum  += f(X, Y, Z) * hx * hy * hz;
                    }
                    ans += sum;
                }
            }
            return ans;
        }, std::plus<double>());
    tbb::tick_count end = tbb::tick_count::now();
    std::cout << "Runtime using TBB: " << static_cast<double>((end - start).seconds()) << std::endl;
    return integral;
}
