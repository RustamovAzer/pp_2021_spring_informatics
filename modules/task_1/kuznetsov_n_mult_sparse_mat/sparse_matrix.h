// Copyright 2021 Kuznetsov Nikita
#ifndef MODULES_TASK_1_KUZNETSOV_N_MULT_SPARSE_MAT_SPARSE_MATRIX_H_
#define MODULES_TASK_1_KUZNETSOV_N_MULT_SPARSE_MAT_SPARSE_MATRIX_H_

#include <vector>

class sMatrix {
 private:
  std::vector<double> val;
  std::vector<int> c_ind, rows;
  int non_zero, size;
 public:
  sMatrix(const std::vector<double>& _val, const std::vector<int>& _c_ind, const std::vector<int>& _rows,
    int _non_zero, int _size);
  sMatrix(int _size, int _non_zero, unsigned int key);
  bool operator==(const sMatrix& mat);
  sMatrix operator*(const sMatrix& mat);
  sMatrix transpose();
};

#endif  // MODULES_TASK_1_KUZNETSOV_N_MULT_SPARSE_MAT_SPARSE_MATRIX_H_
