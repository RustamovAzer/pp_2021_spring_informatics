// Copyright 2021 Volkova Anastasia
#include <iostream>
#include "../../../modules/task_4/volkova_a_rectangle_method/rectangle_method_std.h"
#include "../../../3rdparty/unapproved/unapproved.h"

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
    auto start_calc = std::chrono::high_resolution_clock::now();
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
    auto end_calc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end_calc - start_calc;
    std::cout << "Runtime: " << runtime.count() << std::endl;
    return ans;
}

std::mutex my_mutex;

double ParallelIntegralFunction(double (*f)(double, double, double),
                        double ax, double bx,
                        double ay, double by,
                        double az, double bz,
                        int n, int m, int k) {
    double SUMMA = 0;
    auto start = std::chrono::high_resolution_clock::now();

    const double hx = (bx - ax) / static_cast<double>(n);
    const double hy = (by - ay) / static_cast<double>(m);
    const double hz = (bz - az) / static_cast<double>(k);

    int thread_numb = std::thread::hardware_concurrency();
    std::vector<std::thread> THREADS;
    int div = n / thread_numb;
    int rem = n % thread_numb;
    std::vector<int> end_x(thread_numb);

    for (int i = 0; i < thread_numb; ++i) {
        end_x[i] = div;
        if (rem) {
            rem--;
            end_x[i]++;
        }
    }
    int start_x = 0;

    auto calc_function = [&](int start_x, int end_x){
        double sum = 0;
        double X, Y, Z;
        for (int i = start_x; i < end_x; ++i) {
            X = ax + static_cast<double>(i) * hx + 0.5 * hx;
            for (int j = 0; j < m; ++j) {
                Y = ay + static_cast<double>(j) * hy + 0.5 * hy;
                for (int q = 0; q < k;  ++q) {
                    Z = az + static_cast<double>(q) * hz + 0.5 * hz;
                    sum += f(X, Y, Z);
                }
            }
        }
        std::lock_guard<std::mutex> my_lock(my_mutex);
        SUMMA += sum;
    };

    for (int thread = 0; thread < thread_numb; ++thread) {
        THREADS.emplace_back(std::thread(calc_function,
                            start_x,
                            start_x + end_x[thread]));
        start_x += end_x[thread];
    }
    for (int i = 0; i < thread_numb; ++i) {
        THREADS[i].join();
    }
    SUMMA *= hx * hy * hz;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    std::cout << "Runtime using std::thread : " << runtime.count() << std::endl;
    return SUMMA;
}
