// Copyright 2021 Prokofeva Elizaveta
#ifndef MODULES_TASK_3_PROKOFEVA_E_SPARSE_CRS_TBB_SPARSE_CRS_H_
#define MODULES_TASK_3_PROKOFEVA_E_SPARSE_CRS_TBB_SPARSE_CRS_H_

#include<vector>
using std::vector;

struct crs_matrix {
    int N;
    std::vector<double> values;
    std::vector<int> cols;
    std::vector<int> row_index;
};
crs_matrix transpose(crs_matrix matrix);
crs_matrix mult(crs_matrix first, crs_matrix second);
crs_matrix generate(int size);
crs_matrix mult_tbb(crs_matrix first, crs_matrix second);

#endif  // MODULES_TASK_3_PROKOFEVA_E_SPARSE_CRS_TBB_SPARSE_CRS_H_
