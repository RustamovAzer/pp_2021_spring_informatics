// Copyright 2021 Prokofeva Elizaveta
#include "../../../modules/task_3/prokofeva_e_sparse_crs_tbb/sparse_crs.h"
#include <tbb/tbb.h>
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
        int size = static_cast<int>(result.values.size());
        result.row_index.emplace_back(size);
    }
    return result;
}

crs_matrix transpose(crs_matrix matrix) {
    crs_matrix Tmatrix;
    int N = Tmatrix.N = matrix.N;
    Tmatrix.row_index = vector<int>(static_cast<size_t>(N + 1));
    Tmatrix.values = vector<double>(static_cast<size_t>(matrix.values.size()));
    Tmatrix.cols = vector<int>(static_cast<size_t>(matrix.cols.size()));
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

crs_matrix mult(crs_matrix first, crs_matrix second) {
    crs_matrix res;
    second = transpose(second);
    int N = res.N = first.N;

    vector<int> tmp(N);
    int j, n;
    int tmp1 = 0;

    res.row_index.emplace_back(tmp1);
    for (int i = 0; i < N; i++) {
        tmp.assign(N, -1);

        for (j = first.row_index[i]; j < first.row_index[i + 1]; j++) {
            tmp[first.cols[j]] = j;
        }

        for (j = 0; j < N; j++) {
            double sum = 0;
            int a = second.row_index[j];
            int b = second.row_index[j + 1];
            for (n = a; n < b; n++) {
                int bcol = second.cols[n];
                int ind = tmp[bcol];
                if (ind != -1) {
                    double val1 = first.values[ind];
                    double val2 = second.values[n];
                    sum = sum + val1 * val2;
                }
            }

            if (sum != 0) {
                res.values.emplace_back(sum);
                res.cols.emplace_back(j);
            }
        }
        int size = static_cast<int>(res.values.size());
        res.row_index.emplace_back(size);
    }
    return res;
}

crs_matrix mult_tbb(crs_matrix first, crs_matrix second) {
    crs_matrix res;
    int N = res.N = first.N;
    vector<vector<double>> val(N);
    vector<vector<int>> col(N);
    second = transpose(second);
    int tmp1 = 0;
    res.row_index.emplace_back(tmp1);
    tbb::task_scheduler_init init(8);
    int grainsize = 10;
    tbb::parallel_for(tbb::blocked_range<int>(0, N, grainsize),
        [&](const tbb::blocked_range<int>& r) {
            vector<int> tmp(N);
            for (int k = r.begin(); k < r.end(); k++) {
                tmp.assign(N, -1);
                for (int j = first.row_index[k]; j < first.row_index[k + 1]; j++) {
                    tmp[first.cols[j]] = j;
                }
                for (int i = 0; i < N; i++) {
                    double sum = 0.0;
                    int a = second.row_index[i];
                    int b = second.row_index[i + 1];
                    for (int n = a; n < b; n++) {
                        int bcol = second.cols[n];
                        int ind = tmp[bcol];
                        if (ind != -1) {
                            double val1 = first.values[ind];
                            double val2 = second.values[n];
                            sum = sum + val1 * val2;
                        }
                    }
                    if (sum != 0) {
                        col[k].emplace_back(i);
                        val[k].emplace_back(sum);
                    }
                }
            }
        });
    init.terminate();
    for (int i = 0; i < N; ++i) {
        int size = static_cast<int>(col[i].size());
        res.row_index.emplace_back(res.row_index.back() + size);
        res.values.insert(res.values.end(), val[i].begin(), val[i].end());
        res.cols.insert(res.cols.end(), col[i].begin(), col[i].end());
    }
    return res;
}
