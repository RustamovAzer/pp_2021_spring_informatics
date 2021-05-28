// Copyright 2021 Tashirev Ivan
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <ctime>
#include "../../modules/task_1/tashirev_i_monte_carlo/monte_carlo.h"

double seqMonteCarlo(double(*f)(std::vector<double>), const std::vector<double>& a,
    const std::vector<double>& b, int s) {
    double res = 0.0;
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    if (s <= 0)
        throw "integral is negative";
    int mult = a.size();
    double S = 1;
    for (int i = 0; i < mult; i++)
        S *= (b[i] - a[i]);
    std::vector<std::uniform_real_distribution<double>> r(mult);
    std::vector<double> r1(mult);
    for (int i = 0; i < mult; i++)
        r[i] = std::uniform_real_distribution<double>(a[i], b[i]);
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < mult; j++)
            r1[j] = r[j](gen);
        res += f(r1);
    }
    res *= S / s;
    return res;
}
