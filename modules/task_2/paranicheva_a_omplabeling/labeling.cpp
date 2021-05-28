// Copyright 2021 Paranicheva Alyona
#include <omp.h>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <utility>
#include "../../../modules/task_2/paranicheva_a_omplabeling/labeling.h"

std::vector<int> getRandomMatrix(int rows, int cols) {
    if ((rows <= 0) || (cols <= 0))
        throw std::invalid_argument("Matrix can't be created");
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> array(rows * cols);
    for (int i = 0; i < cols; i++)
        array[i] = 0;
    for (int i = 1; i < rows - 1; i++)
        for (int j = 0; j < cols; j++) {
            if ((j == 0) || (j == cols - 1))
                array[i * cols + j] = 0;
            else
                array[i * cols + j] = static_cast<int>(gen() % 2);
        }
    for (int i = 0; i < cols; i++)
        array[(rows - 1) * cols + i] = 0;
    return array;
}

std::pair<std::vector<int>, std::vector<int> > firstMark(std::vector<int> arr, int rows, int cols) {
    std::vector<int> sets(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        sets[i] = i;
    for (int i = 1; i < rows - 1; i++)
        for (int j = 1; j < cols - 1; j++) {
            if (arr[i * cols + j] == 0)
                continue;
            if ((arr[i * cols + j - 1] == 0) && (arr[(i - 1) * cols + j] == 0)) {
                arr[i * cols + j] = i * cols + j + 1;
                continue;
            }
            if ((arr[i * cols + j - 1] != 0) && (arr[(i - 1) * cols + j] == 0)) {
                arr[i * cols + j] = arr[i * cols + j - 1];
                continue;
            }
            if ((arr[i * cols + j - 1] == 0) && (arr[(i - 1) * cols + j] != 0)) {
                arr[i * cols + j] = arr[(i - 1) * cols + j];
                continue;
            }
            if ((arr[i * cols + j - 1] != 0) && (arr[(i - 1) * cols + j] != 0)) {
                int max, min;
                if (arr[i * cols + j - 1] < arr[(i - 1) * cols + j]) {
                    max = arr[(i - 1) * cols + j];
                    min = arr[i * cols + j - 1];
                } else {
                    max = arr[i * cols + j - 1];
                    min = arr[(i - 1) * cols + j];
                }
                while (sets[max] != max) {
                    max = sets[max];
                }
                while (sets[min] != min)
                    min = sets[min];
                if (min != max)
                    sets[max] = min;
                arr[i * cols + j] = min;
            }
        }
    return { arr, sets };
}

std::vector<int> secondMark(std::vector<int> arr, int rows, int cols, std::vector<int> sets) {
    std::vector<int> res(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        res[i] = arr[i];
    for (int i = cols; i < (rows - 1) * cols; i++) {
        if (arr[i] == 0)
            continue;
        if (sets[arr[i]] == arr[i]) {
            res[i] = arr[i];
        } else {
            while (sets[arr[i]] != arr[i])
                arr[i] = sets[arr[i]];
            res[i] = arr[i];
        }
    }
    return res;
}

std::vector<int> CLabeling(const std::vector<int>& array, int rows, int cols) {
    std::vector<int> res(rows * cols);
    std::pair<std::vector<int>, std::vector<int> > firstm = firstMark(array, rows, cols);
    std::vector<int> arr = firstm.first;
    std::vector<int> sets = firstm.second;
    res = secondMark(arr, rows, cols, sets);
    return Transform(res, rows, cols);
}

std::vector<int> CLabelingOmp(const std::vector<int>& array, int rows, int cols) {
    std::vector<int> res(rows * cols);
    std::pair<std::vector<int>, std::vector<int> > firstm = firstMarkOmp(array, rows, cols);
    std::vector<int> arr = firstm.first;
    std::vector<int> sets = firstm.second;
    res = secondMark(arr, rows, cols, sets);
    return Transform(res, rows, cols);
}

std::pair<std::vector<int>, std::vector<int> > firstMarkOmp(std::vector<int> arr, int rows, int cols) {
    int threads = 4;
    std::vector<int> sets(rows * cols);
    std::vector<int> strbeg(threads);
    int num = (rows - 2) / threads;
    int rem = (rows - 2) % threads;
    for (int i = 0; i < threads; i++) {
        strbeg[i] = i * num + 1;
    }
    std::vector<int> kolvo(threads);
    for (int i = 0; i < threads; i++) {
        if (i == threads - 1)
            kolvo[i] = num + rem;
        else
            kolvo[i] = num;
    }
    for (int i = 0; i < rows * cols; i++)
        sets[i] = i;

#pragma omp parallel for shared(sets, arr) num_threads(threads)
    for (int th = 0; th < threads; th++) {
        for (int i = strbeg[th]; i < strbeg[th] + kolvo[th]; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if (arr[i * cols + j] == 0)
                    continue;
                if ((arr[i * cols + j - 1] == 0) && ((arr[(i - 1) * cols + j] == 0) || (i == strbeg[th]))) {
                    arr[i * cols + j] = i * cols + j + 1;
                        continue;
                }
                if ((arr[i * cols + j - 1] != 0) && ((arr[(i - 1) * cols + j] == 0) || (i == strbeg[th]))) {
                    arr[i * cols + j] = arr[i * cols + j - 1];
                        continue;
                }
                if ((arr[i * cols + j - 1] == 0) && (arr[(i - 1) * cols + j] != 0)) {
                    arr[i * cols + j] = arr[(i - 1) * cols + j];
                        continue;
                }
                if ((arr[i * cols + j - 1] != 0) && (arr[(i - 1) * cols + j] != 0)) {
                    int max, min;
                    if (arr[i * cols + j - 1] < arr[(i - 1) * cols + j]) {
                        max = arr[(i - 1) * cols + j];
                        min = arr[i * cols + j - 1];
                    } else {
                        max = arr[i * cols + j - 1];
                        min = arr[(i - 1) * cols + j];
                    }
                    while (sets[max] != max) {
                        max = sets[max];
                    }
                    while (sets[min] != min)
                        min = sets[min];
                    if (min != max)
                        sets[max] = min;
                    arr[i * cols + j] = min;
                }
            }
        }
    }
    for (int i = 1; i < threads; i++) {
        int str = strbeg[i];
        for (int j = 0; j < cols; j++) {
            if ((arr[(str - 1) * cols + j] != 0) && (arr[str * cols + j] != 0)) {
                int first = sets[arr[(str - 1) * cols + j]];
                int second = sets[arr[str * cols + j]];
                if (first != second) {
                    int max;
                    if (first > second)
                        max = first;
                    else
                        max = second;
                    while (sets[max] != max)
                        max = sets[max];
                    int min;
                    if (first < second)
                        min = first;
                    else
                        min = second;
                    while (sets[min] != min)
                        min = sets[min];
                    if (max != min)
                        sets[max] = min;
                }
            }
        }
    }
    return { arr, sets };
}

std::vector<int> Transform(const std::vector<int>& array, int rows, int cols) {
    std::vector<int> res(rows * cols, 0);
    std::vector<int> count;
    for (int i = 0; i < rows * cols; i++) {
        if (array[i] != 0) {
            bool flag = true;
            for (uint32_t j = 0; j < count.size(); j++)
                if (count[j] == array[i])
                    flag = false;
            if (flag)
                count.push_back(array[i]);
        }
    }
    for (int i = 0; i < rows * cols; i++) {
        if (array[i] != 0) {
            int flag = 0;
            for (uint32_t j = 0; j < count.size(); j++)
                if (count[j] == array[i])
                    flag = j + 1;
            res[i] = flag;
        }
    }
    return res;
}
