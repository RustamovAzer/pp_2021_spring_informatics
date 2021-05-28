// Copyright 2021 Bessolytsin Sergey
#ifndef MODULES_TASK_1_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_
#define MODULES_TASK_1_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_

#include <vector>
#include <random>
#include <utility>

template<typename T>
void quick_sort(T arr[], int right) {
    if (right > 0) {
        int rand_id = std::rand() % right;  // Change to std::mt19937?
        int k = 0;
        std::swap(arr[0], arr[rand_id]);
        for (int i = 1; i < right; ++i) {
            if (arr[i] < arr[0]) {
                std::swap(arr[i], arr[++k]);
            }
        }
        std::swap(arr[0], arr[k]);
        quick_sort(arr, k);
        quick_sort(arr + k + 1, right - k - 1);
    }
}

std::vector<int> getRandomVector(int size, uint64_t seed = 50);
std::vector<double> getRandomDoubleVector(int size, uint64_t seed = 50);

#endif  // MODULES_TASK_1_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_
