// Copyright 2021 Paranicheva Alyona
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include "../../../modules/task_1/paranicheva_a_labeling/labeling.h"

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
    int obj = 0;
    std::vector<int> rel;
    rel.push_back(0);
    for (int i = 1; i < rows - 1; i++)
        for (int j = 1; j < cols - 1; j++) {
            if (arr[i * cols + j] == 0)
                continue;
            if ((arr[i * cols + j - 1] == 0) && (arr[(i - 1) * cols + j] == 0)) {
                obj++;
                rel.push_back(obj);
                arr[i * cols + j] = obj;
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
                int tmp, tmpch;
                if (rel[arr[i * cols + j - 1]] < rel[arr[(i - 1) * cols + j]]) {
                    tmp = rel[arr[i * cols + j - 1]];
                    tmpch = rel[arr[(i - 1) * cols + j]];
                } else {
                    tmp = rel[arr[(i - 1) * cols + j]];
                    tmpch = rel[arr[i * cols + j - 1]];
                }
                arr[i * cols + j] = tmp;
                rel[tmpch] = tmp;
                for (uint32_t k = 0; k < rel.size(); k++)
                    if (rel[k] == tmpch)
                        rel[k] = tmp;
            }
        }
    return { arr, rel };
}

std::vector<int> secondMark(std::vector<int> arr, int rows, int cols, std::vector<int> rel) {
    for (int i = cols; i < (rows - 1) * cols; i++) {
        if (arr[i] == 0)
            continue;
        arr[i] = rel[arr[i]];
    }
    return arr;
}

std::vector<int> CLabeling(const std::vector<int>& array, int rows, int cols) {
    std::vector<int> res(rows * cols);
    std::pair<std::vector<int>, std::vector<int> > firstm = firstMark(array, rows, cols);
    std::vector<int> arr = firstm.first;
    std::vector<int> rel = firstm.second;
    res = secondMark(arr, rows, cols, rel);
    return res;
}
