// Copyright 2021 Prokofeva Elizaveta
#ifndef MODULES_TASK_1_PROKOFEVA_E_SPARSE_CRS_SPARSE_CRS_H_
#define MODULES_TASK_1_PROKOFEVA_E_SPARSE_CRS_SPARSE_CRS_H_

#include<vector>

struct crs_matrix {
    int N;
    std::vector<double> values;
    std::vector<int> cols;
    std::vector<int> row_index; };

crs_matrix transpose(crs_matrix matrix);
crs_matrix create(int size, std::vector<double> matrix);
crs_matrix mult(crs_matrix first, crs_matrix second);
double scalar_mult(crs_matrix first, crs_matrix second, int i, int j);

#endif  // MODULES_TASK_1_PROKOFEVA_E_SPARSE_CRS_SPARSE_CRS_H_
