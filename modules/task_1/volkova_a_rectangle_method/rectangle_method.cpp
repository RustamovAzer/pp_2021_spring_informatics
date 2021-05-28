// Copyright 2021 Volkova Anastasia
#include <iostream>
#include "../../../modules/task_1/volkova_a_rectangle_method/rectangle_method.h"

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
    const double hx = (bx - ax) / static_cast<double>(n);
    const double hy = (by - ay) / static_cast<double>(m);
    const double hz = (bz - az) / static_cast<double>(k);

    double ans = 0;
    double X, Y, Z;
    for (double i = 0; i < n; ++i) {
        X = static_cast<double>(ax) + i * hx + 0.5 * hx;
        for (double j = 0; j < m; ++j) {
            Y = static_cast<double>(ay) + j * hy + 0.5 * hy;
            for (double q = 0; q < k;  ++q) {
                Z = static_cast<double>(az) + q * hz + 0.5 * hz;
                ans += f(X, Y, Z);
            }
        }
    }
    ans *= hx * hy * hz;
    return ans;
}
