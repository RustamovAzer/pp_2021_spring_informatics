// Copyright 2021 Streltsova Yana
#ifndef MODULES_TASK_2_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
#define MODULES_TASK_2_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_

#include <omp.h>

#include <utility>
#include <vector>

std::vector<int> getRandomVectorInt(int  sz);

std::vector<double> getRandomVectorDouble(int  sz);

template< typename T >
std::vector<T> shell_sort_sequential(const std::vector<T>& vec) {
    std::vector<T> result(vec);
    int d = 4;
    while (d > 0) {
        for (size_t i = 0; i < result.size(); i++) {
            for (size_t j = i + d; j < result.size(); j += d) {
                if (result[i] > result[j])
                    std::swap(result[i], result[j]);
            }
        }
        d /= 2;
    }
    return result;
}

template< typename T >
std::vector<T> shell_sort_parallel(const std::vector<T>& vec) {
    if (vec.size() < 1000)
        return shell_sort_sequential(vec);
    std::vector<T> result(vec);
    int d = 4;
    while (d > 0) {
        omp_set_num_threads(d);
#pragma omp parallel
        {
        int tid = omp_get_thread_num();
        for (size_t i = tid; i < result.size(); i += d)
            for (size_t j = i + d; j < result.size(); j += d) {
                if (result[i] > result[j])
                    std::swap(result[i], result[j]);
            }
        }
        d /= 2;
    }
    return result;
}

#endif  // MODULES_TASK_2_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
