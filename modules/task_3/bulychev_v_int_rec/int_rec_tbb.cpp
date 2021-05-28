// Copyright 2021 Bulychev Vladislav
#include <tbb/tbb.h>
#include <vector>
#include <numeric>
#include <utility>
#include <cmath>
#include <functional>
#include "../../../modules/task_3/bulychev_v_int_rec/int_rec_tbb.h"

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double)) {
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
        result += f(p[0], p[1]);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}

double Calculation_Tbb(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double)) {
    tbb::task_scheduler_init init(1);
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

    result = tbb::parallel_reduce(
        tbb::blocked_range<int> {0, num}, 0.f,
        [&](const tbb::blocked_range<int>& r, double l_result)->double {
            int begin = r.begin();
            int end = r.end();
            for (int i = begin; i < end; i++) {
                for (int j = 0; j < size_a; j++) {
                    p[j] = (i % n) * h[j] + a[j] + h[j] * 0.5;
                }
                l_result += f(p[0], p[1]);
            }
            return l_result;
        }, std::plus<double>());

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}
