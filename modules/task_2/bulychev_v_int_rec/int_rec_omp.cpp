// Copyright 2021 Bulychev Vladislav
#include <omp.h>
#include <vector>
#include <numeric>
#include <utility>
#include <cmath>
#include <functional>
#include "../../../modules/task_2/bulychev_v_int_rec/int_rec_omp.h"

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>)) {
    int size_a = a.size();
    std::vector<double> h;
    double result = 0.0;
    std::vector <double> p(size_a);
    int num = pow(n, size_a);

    for (int i = 0; i < size_a; i++) {
        double t1 = b[i] - a[i];
        double t2 = t1 / n;
        h.push_back(t2);
    }

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < size_a; j++) {
            double t3 = h[j] * 0.5;
            p[j] = (i % n) * h[j] + a[j] + t3;
        }
        result += f(p);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}

double Calculation_Omp(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>)) {
    int size_a = a.size();
    std::vector<double> h;
    int num = pow(n, size_a);

    for (int i = 0; i < size_a; i++) {
        double t1 = b[i] - a[i];
        double t2 = t1 / n;
        h.push_back(t2);
    }

    double result = 0.0;
    double r = 0.0;
    std::vector <double> p(size_a);

#pragma omp parallel for reduction(+ : r)
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < size_a; j++) {
            double t3 = h[j] * 0.5;
            p[j] = (i % n) * h[j] + a[j] + t3;
        }
        r += f(p);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = r * t5;

    return result;
}
