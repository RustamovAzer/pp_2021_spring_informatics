// Copyright 2021 Kuznetsov Nikita
#ifndef MODULES_TASK_2_KUZNETSOV_N_MULT_SPARSE_MAT_OMP_SPARSE_MATRIX_H_
#define MODULES_TASK_2_KUZNETSOV_N_MULT_SPARSE_MAT_OMP_SPARSE_MATRIX_H_

#include <vector>

class sMatrix {
 public:
  std::vector<double> val;
 private:
  std::vector<int> c_ind, rows;
  int non_zero, size;
 public:
  sMatrix(const std::vector<double>& _val, const std::vector<int>& _c_ind, const std::vector<int>& _rows,
    int _non_zero, int _size);
  sMatrix(int _size, int _non_zero, unsigned int key);
  bool operator==(const sMatrix& mat);
  sMatrix operator*(const sMatrix& mat);
  sMatrix transpose();
  sMatrix pMult(const sMatrix& mat, int threads);
};

#endif  // MODULES_TASK_2_KUZNETSOV_N_MULT_SPARSE_MAT_OMP_SPARSE_MATRIX_H_
