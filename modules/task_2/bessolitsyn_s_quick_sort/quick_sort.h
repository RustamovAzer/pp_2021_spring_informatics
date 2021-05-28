// Copyright 2021 Bessolitsyn Sergey
#ifndef MODULES_TASK_2_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_
#define MODULES_TASK_2_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_

#include <omp.h>

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

template<typename T>
void merge(T arr1[], int size1, T arr2[], int size2) {
    T* tmp_arr = new T[size1 + size2];
    int i = 0, j = 0, k = 0;
    for (; i < size1 && j < size2; ++k) {
        if (arr1[i] < arr2[j])
            tmp_arr[k] = arr1[i++];
        else
            tmp_arr[k] = arr2[j++];
    }
    while (i < size1) {
        tmp_arr[k++] = arr1[i++];
    }
    while (j < size2) {
        tmp_arr[k++] = arr2[j++];
    }
    for (i = 0; i < k; ++i) {
        arr1[i] = tmp_arr[i];
    }
}


template<typename T>
void quick_sort_OMP(T arr[], int size) {
    #pragma omp parallel
    {
        int num_th = omp_get_num_threads();
        int delta = size / num_th;
        int rem = size % num_th;
        int th_id = omp_get_thread_num();
        if (th_id != num_th - 1)
            quick_sort(arr + th_id * delta, delta);
        else
            quick_sort(arr + (num_th - 1) * delta, delta + rem);
        #pragma omp barrier
        #pragma omp master
        {
            int last_size = delta + rem;
            T* arr2 = arr + (num_th - 1) * delta;
            T* arr1 = arr2 - delta;
            for (int j = num_th - 1; j > 0; --j) {
                merge(arr1, delta, arr2, last_size);
                arr1 -= delta;
                arr2 -= delta;
                last_size += delta;
            }
        }
    }
}


std::vector<int> getRandomVector(int size, uint64_t seed = 50);
std::vector<double> getRandomDoubleVector(int size, uint64_t seed = 50);

#endif  // MODULES_TASK_2_BESSOLITSYN_S_QUICK_SORT_QUICK_SORT_H_
