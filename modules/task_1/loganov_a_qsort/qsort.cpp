// Copyright 2021 Loganov Andrei
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>
#include "../../../../modules/task_1/loganov_a_qsort/qsort.h"
std::vector<double> copyvec(std::vector<double> vec) {
    std::vector<double> result(vec.size());
    for (int i=0; i<static_cast<int>(vec.size()); i++) {
        result[i] = vec[i];
    }
    return result;
}
std::vector<double> getRandomVector(int size) {
    std::mt19937 gen;
    std::vector<double> result(size);
    const int start = 0;
    const int end = 1000;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<double> rand(start, end);
    for (int i = 0; i < static_cast<int>(result.size()); i++) {
        result[i] = rand(gen);
    }
    return result;
}
void Hoarsort(std::vector<double>* vec, int first, int last) {
    if (first >= last)return;
    int i = first, j = last;
    double  mid = (*vec)[(last + first) / 2];
    while (i <= j) {
        while ((*vec)[i] < mid) i++;
        while ((*vec)[j] > mid)j--;
        if (i < j) std::swap((*vec)[i], (*vec)[j]);
        if (i <= j) {
            i++;
            j--;
        }
    }
    Hoarsort(vec, first, j);
    Hoarsort(vec, i, last);
}
