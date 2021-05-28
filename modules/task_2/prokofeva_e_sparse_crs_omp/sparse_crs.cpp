// Copyright 2021 Prokofeva Elizaveta
#include "../../../modules/task_2/prokofeva_e_sparse_crs_omp/sparse_crs.h"
#include <omp.h>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>

crs_matrix generate(int size) {
    crs_matrix result;
    result.N = size;
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<double> dis(-100, 100);
    result.row_index.emplace_back(0);
    for (int i = 0; i < size; i++) {
        result.values.emplace_back(dis(gen));
        result.cols.emplace_back(gen() % size);
        result.row_index.emplace_back(static_cast<int>(result.values.size()));
    }
    return result;
}

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
                int v1 = first.values[k];
                int v2 = second.values[l];
                summ = summ + v1 * v2;
                break;
            }
        }
    }
    return summ;
}

crs_matrix mult(crs_matrix first, crs_matrix second) {
    crs_matrix result;
    second = transpose(second);
    double summ;

    result.row_index.push_back(0);
    for (int i = 0; i < first.N; i++) {
        for (int j = 0; j < second.N; j++) {
            summ = scalar_mult(first, second, i, j);
            if (summ != 0) {
                result.values.push_back(summ);
                result.cols.push_back(j);
            }
        }
        result.row_index.push_back(static_cast<int>(result.values.size()));
    }
    return result;
}

crs_matrix parallel_mult(crs_matrix first, crs_matrix second) {
    crs_matrix res;
    int N = res.N = first.N;
    std::vector<std::vector<double>> val(N);
    std::vector<std::vector<int>> col(N);
    std::vector<int> rind;
    second = transpose(second);
    res.row_index.emplace_back(0);
    int n, j;
#pragma omp parallel
    {
        std::vector<int> tmp(N);
        #pragma omp for private(j, n) schedule(static)
        for (int i= 0; i < N; i++) {
            tmp.assign(N, -1);

            for (j = first.row_index[i]; j < first.row_index[i + 1]; j++) {
                int coln = first.cols[j];
                tmp[coln] = j;
            }

            for (j = 0; j < N; j++) {
                double sum = 0.0;
                int a = second.row_index[j];
                int b = second.row_index[j + 1];
                for (n = a; n < b; n++) {
                    int bcol = second.cols[n];
                    int ind = tmp[bcol];
                    if (ind != -1) {
                        int v1 = first.values[ind];
                        int v2 = second.values[n];
                        sum = sum + v1 * v2;
                    }
                }
                if (sum != 0) {
                    val[i].emplace_back(sum);
                    col[i].emplace_back(j);
                }
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        int size = static_cast<int>(col[i].size());
        res.row_index.push_back(res.row_index.back() + size);
        res.values.insert(res.values.end(), val[i].begin(), val[i].end());
        res.cols.insert(res.cols.end(), col[i].begin(), col[i].end());
    }
    return res;
}
