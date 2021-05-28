// Copyright 2021 Taktaev Artem
#include "../../../modules/task_1/taktaev_a_qsort/qsort.h"

#include <random>

std::vector<double> createRandomVector(int vec_size) {
    if (vec_size <= 0) throw "Vector's size must be > 0.";

    std::random_device rand_d;
    std::mt19937 gen(rand_d());
    std::vector<double> res_vec(vec_size);

    for (int  i = 0; i < vec_size; i++)
        res_vec[i] = static_cast<double>((gen() % 101) - 50);

    return res_vec;
}

void qSort(std::vector<double> *arr, int left, int right) {
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

    if (j > left) qSort(arr, left, j);
    if (i < right) qSort(arr, i, right);
}
