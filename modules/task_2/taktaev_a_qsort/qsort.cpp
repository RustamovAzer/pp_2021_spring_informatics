// Copyright 2021 Taktaev Artem
#include "../../../modules/task_2/taktaev_a_qsort/qsort.h"

#include <omp.h>

#include <random>
#include <utility>
#include <deque>

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
    if (left >= right) throw "Left idx must be >= that right one.";

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
    if (size_a < 1) throw "Size of a must be > 0";
    if (size_b < 1) throw "Size of b must be > 0";
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

void qSortOmp(std::vector<double> *arr) {
    #pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int part_vec_size = static_cast<int>(arr->size()) / n_threads;
            int thread_id = omp_get_thread_num();
            if (thread_id != n_threads - 1) {
                qSortSeq(arr, thread_id * part_vec_size, (thread_id + 1) * part_vec_size - 1);
            } else {
                qSortSeq(arr, thread_id * part_vec_size, static_cast<int>(arr->size()) - 1);
            }
    }

    #pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        #pragma omp master
        {
            int part_vec_size = static_cast<int>(arr->size()) / n_threads;
            std::deque<std::pair<int, int>> idx_size_deque;
            for (int i = 0; i < n_threads - 1; i++) {
                idx_size_deque.push_back({i, part_vec_size});
            }
            int n_last = static_cast<int>(arr->size()) - (n_threads - 1) * part_vec_size;
            idx_size_deque.push_back({n_threads - 1, n_last});
            std::pair<int, int> arr_to_merge1;
            std::pair<int, int> arr_to_merge2;
            while (idx_size_deque.size() > 1) {
                arr_to_merge1 = idx_size_deque.front();
                idx_size_deque.pop_front();
                arr_to_merge2 = idx_size_deque.front();
                idx_size_deque.pop_front();
                merge(arr->data() + arr_to_merge1.first * part_vec_size, arr_to_merge1.second,
                 arr->data() + arr_to_merge2.first * part_vec_size, arr_to_merge2.second);
                idx_size_deque.push_front({arr_to_merge1.first, arr_to_merge1.second + arr_to_merge2.second});
            }
        }
    }
}
