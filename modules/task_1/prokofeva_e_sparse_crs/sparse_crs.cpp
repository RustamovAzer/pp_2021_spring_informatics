// Copyright 2021 Prokofeva Elizaveta
#include "../../../modules/task_1/prokofeva_e_sparse_crs/sparse_crs.h"
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

crs_matrix create(int size, std::vector<double> matrix) {
    crs_matrix result;
    result.N = size;
    int tmp = 0;

    result.row_index.push_back(tmp);
    for (int i = 0; i < size; i++) {
        int count = 0;
        for (int j = 0; j < size; j++) {
            if (matrix[result.N * i + j] != 0) {
                count++;
                result.cols.push_back(j);
                result.values.push_back(matrix[result.N * i + j]);
            }
        }
        count += result.row_index.back();
        result.row_index.push_back(result.cols.size());
    }

    return result;
}

crs_matrix transpose(crs_matrix matrix) {
    crs_matrix Tmatrix;
    int N = Tmatrix.N = matrix.N;
    Tmatrix.row_index = std::vector<int>(static_cast<size_t>(N + 1));
    Tmatrix.values = std::vector<double>(static_cast<size_t>(matrix.values.size()));
    Tmatrix.cols = std::vector<int>(static_cast<size_t>(matrix.cols.size()));
    size_t val = matrix.values.size();
    for (size_t i = 0; i < val; i++) {
        int col = matrix.cols[i];
        Tmatrix.row_index[col + 1]++;
    }

    double s = 0;
    for (int i = 1; i <= N; i++) {
        int temp = Tmatrix.row_index[i];
        Tmatrix.row_index[i] = s;
        s += temp;
    }

    for (int i = 0; i < N; i++) {
        for (int j = matrix.row_index[i]; j < matrix.row_index[i + 1]; j++) {
            Tmatrix.values[Tmatrix.row_index[matrix.cols[j] + 1]] = matrix.values[j];
            Tmatrix.cols[Tmatrix.row_index[matrix.cols[j] + 1]] = i;
            Tmatrix.row_index[matrix.cols[j] + 1]++;
        }
    }
    return Tmatrix;
}

double scalar_mult(crs_matrix first, crs_matrix second, int i, int j) {
    double summ = 0.0;
    int a = first.row_index[i];
    int a1 = first.row_index[i + 1];
    int b = second.row_index[j];
    int b1 = second.row_index[j + 1];
    for (int k = a; k < a1; k++) {
        for (int l = b; l < b1; l++) {
            if (first.cols[k] == second.cols[l]) {
                summ = summ + first.values[k] * second.values[l];
                break;
            }
        }
    }
    return summ;
}

crs_matrix mult(crs_matrix first, crs_matrix second) {
    int N = first.N;
    crs_matrix result;
    second = transpose(second);
    double summ;
    int rowNZ = 0;
    int tmp = 0;

    result.row_index.emplace_back(tmp);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            summ = scalar_mult(first, second, i, j);
            if (summ != 0) {
                rowNZ++;
                result.cols.emplace_back(j);
                result.values.emplace_back(summ);
            }
        }
        result.row_index.emplace_back(result.row_index[i] + rowNZ);
    }
    return result;
}
