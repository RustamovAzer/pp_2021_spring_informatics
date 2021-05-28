// Copyright 2021 Taktaev Artem
#include "../../../modules/task_3/taktaev_a_qsort/qsort.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <map>
#include <random>
#include <utility>

std::vector<double> createRandomVector(int vec_size) {
    if (vec_size <= 0) throw "Vector's size must be > 0.";

    std::random_device rand_d;
    std::mt19937 gen(rand_d());
    std::vector<double> res_vec(vec_size);

    for (int i = 0; i < vec_size; i++)
        res_vec[i] = static_cast<double>(gen() % 2001) - 1000.0;

    return res_vec;
}

void qSortSeq(std::vector<double> *arr, int left, int right) {
    if (left >= right) throw "Left idx must be < that right one.";

    int pidx = (left + right) / 2;
    double p = arr->at(pidx);
    int i = left, j = right;

    do {
        while (arr->at(i) < p) i++;
        while (arr->at(j) > p) j--;
        if (i <= j) {
            if (i < j) std::swap(arr->at(i), arr->at(j));
            i++;
            j--;
        }
    } while (i <= j);

    if (j > left) qSortSeq(arr, left, j);
    if (i < right) qSortSeq(arr, i, right);
}

void merge(double* a, int size_a, double* b, int size_b) {
    if (size_a < 1 || size_b < 1) return;
    int i = 0, j = 0, k = 0;
    int size_c = size_a + size_b;
    double* c = new double[size_c];
    while ((i < size_a) && (j < size_b)) {
        if (a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }

    if (i == size_a) {
        while (j < size_b) {
            c[k++] = b[j++];
        }
    } else {
        while (i < size_a) {
            c[k++] = a[i++];
        }
    }

    for (int i = 0; i < size_c; i++) {
        a[i] = c[i];
    }
}

void qSortTbb(std::vector<double> *arr) {
    int size = static_cast<int>(arr->size());
    int min_grain_size = 10;  // the less possible grain size
    int divider = 3;  // the part of arr that can be a grain size
    int part_arr_size = size / divider;
    int grain_size = part_arr_size > min_grain_size ? part_arr_size : min_grain_size;
    tbb::parallel_for(tbb::blocked_range<int>(0, size, grain_size), [&arr](const tbb::blocked_range<int>& r) {
        qSortSeq(arr, r.begin(), r.end() - 1);
    });
    int i_prev = 0;
    int k = 0;
    for (int i = 1; i < size; i++) {
        if (arr->at(i) < arr->at(i - 1)) {
            if (k++ > 0)
                merge(arr->data(), i_prev, arr->data() + i_prev, i - i_prev);
            i_prev = i;
        }
    }
    merge(arr->data(), i_prev, arr->data() + i_prev, size - i_prev);
}
