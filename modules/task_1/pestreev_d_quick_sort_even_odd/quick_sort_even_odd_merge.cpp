// Copyright 2021 Pestreev Daniil
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <utility>
#include <ctime>
#include "../../../modules/task_1/pestreev_d_quick_sort_even_odd/quick_sort_even_odd_merge.h"

std::vector<int> getRandomVector(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> a(size);
    for (int i = 0; i < size; i++) {
        a[i] = gen();
    }
    return a;
}

void qsort(int* vec, int left, int right) {
    int mid;
    int l = left;
    int r = right;
    mid = vec[(l + r) / 2];
    do {
        while (vec[l] < mid) l++;
        while (vec[r] > mid) r--;
        if (l <= r) {
            std::swap(vec[l], vec[r]);
            l++;
            r--;
        }
    } while (l < r);
    if (left < r) qsort(vec, left, r);
    if (l < right) qsort(vec, l, right);
}

std::vector<int> quickSortV(const std::vector<int>& vec) {
    int size = vec.size();
    if (size <= 1) {
        return vec;
    }
    int* arr = new int[size];
    for (int i = 0; i < size; i++) {
        arr[i] = vec[i];
    }
    qsort(arr, 0, size - 1);
    std::vector<int> tmp;
    for (int i = 0; i < size; i++) {
        tmp.push_back(arr[i]);
    }
    delete[] arr;
    return tmp;
}
