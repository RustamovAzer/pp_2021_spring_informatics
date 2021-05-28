// Copyright 2021 Taktaev Artem
#include "../../../modules/task_4/taktaev_a_qsort/qsort.h"

#include <random>
#include <utility>
#include <vector>

#include "../../../3rdparty/unapproved/unapproved.h"

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
    delete[] c;
}

void qSortStd(std::vector<double> *arr, int thread_count) {
    std::vector<std::thread> threads;
    int part_vec_size = static_cast<int>(arr->size()) / thread_count;
    for (int i = 0; i < thread_count; i++) {
        std::thread th([&arr, &thread_count, &part_vec_size] (int i) {
            if (i != thread_count - 1) {
                qSortSeq(arr, i * part_vec_size, (i + 1) * part_vec_size - 1);
            } else {
                qSortSeq(arr, i * part_vec_size, static_cast<int>(arr->size()) - 1);
            }
        }, i);
        threads.emplace_back(std::move(th));
    }

    for (int i = 0; i < thread_count; i++)
        threads[i].join();

    for (int i = 1; i < thread_count; i++) {
        int current_size = part_vec_size;
        if (i == thread_count - 1) {
            current_size = static_cast<int>(arr->size()) - (thread_count - 1) * part_vec_size;
        }
        merge(arr->data(), part_vec_size * i, arr->data() + part_vec_size * i, current_size);
    }
}
