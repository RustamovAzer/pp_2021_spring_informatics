// Copyright 2021 Tashirev Ivan
#include <omp.h>
#include <ctime>
#include <random>
#include <string>
#include <algorithm>
#include <vector>
#include "../../../modules/task_2/tashirev_i_monte_carlo/monte_carlo.h"

double seqMonteCarlo(double(*f)(const std::vector<double>&),
                                const std::vector<double>& a,
                                const std::vector<double>& b, int steps) {
    if (steps <= 0)
        throw "integral is negative";
    double res = 0.0;
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int mult = a.size();
    double S = 1;
    for (int i = 0; i < mult; i++)
        S *= (b[i] - a[i]);

    std::vector<std::uniform_real_distribution<double>> r(mult);
    std::vector<double> r1(mult);
    for (int i = 0; i < mult; i++)
        r[i] = std::uniform_real_distribution<double>(a[i], b[i]);

    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < mult; ++j)
            r1[j] = r[j](gen);
        res += f(r1);
    }

    res *= S / steps;
    return res;
}

double ompMonteCarlo(double(*f)(const std::vector<double>&),
                                const std::vector<double>& a,
                                const std::vector<double>& b, int steps) {
    if (steps <= 0)
        throw "integral is negative";
    double res = 0.0;
    int mult = a.size();
    std::vector<std::uniform_real_distribution<double>> r(mult);
    for (int i = 0; i < mult; i++)
        r[i] = std::uniform_real_distribution<double>(a[i], b[i]);
    int num_th;
    #pragma omp parallel shared(r) reduction(+ : res)
    {
        std::mt19937 gen;
        gen.seed(static_cast<unsigned int>(time(0)));
        num_th = omp_get_num_threads();
        std::vector<double> r1(mult);
        for (int i = 0; i < steps; ++i) {
            for (int j = 0; j < mult; ++j)
                r1[j] = r[j](gen);
            res += f(r1);
        }
    }
    #pragma omp master
    {
        double S = 1;
        for (int i = 0; i < mult; i++)
            S *= (b[i] - a[i]);
        res *= S / (num_th * steps);
    }
    return res;
}
