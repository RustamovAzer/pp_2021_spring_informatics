// Copyright 2021 Ludina Daria
#include "../../../modules/task_1/ludina_d_sparse_matrix_complex_ccs/sparse_matrix_complex_ccs.h"

SparseMatrix::SparseMatrix() {
  rows = 0;
  cols = 0;
}

SparseMatrix::SparseMatrix(int _rows, int _cols, std::vector<std::complex<int>> matrix) {
  rows = _rows;
  cols = _cols;
  int not_zero_value = 0;
  col_idx.push_back(0);
  for (int j = 0; j < _cols; j++) {
    for (int i = 0; i < _rows; i++) {
      if (matrix[cols * i + j] != std::complex<int>(0, 0)) {
        not_zero_value++;
        value.push_back(matrix[cols * i + j]);
        row.push_back(i);
      }
    }
    col_idx.push_back(not_zero_value);
  }
}

SparseMatrix SparseMatrix::transpose() {
  std::vector<std::complex<int>> valueT(value.size());
  std::vector<int> col_idxT(col_idx.size());
  std::vector<int> rowT(row.size());
  int size = row.size();
  for (int i = 0; i < size; i++) {
    col_idxT[row[i] + 1]++;
  }
  int tmp, index, r;
  int s = 0;
  for (int i = 1; i <= rows; i++) {
    tmp = col_idxT[i];
    col_idxT[i] = s;
    s += tmp;
  }
  for (int i = 0; i < cols; i++) {
    for (int j = col_idx[i]; j < col_idx[i + 1]; ++j) {
      r = row[j];
      index = col_idxT[r + 1];
      rowT[index] = i;
      valueT[index] = value[j];
      col_idxT[r + 1]++;
    }
  }
  SparseMatrix res = SparseMatrix(rows, cols, col_idxT, rowT, valueT);
  return res;
}

SparseMatrix Multiplication(SparseMatrix A, SparseMatrix B) {
  if (A.cols != B.rows)
    throw "Size col A not equal size row B";
  SparseMatrix result;
  A.transpose();
  int col_not_zero = 0;
  std::vector<std::complex<int>> tmp(A.rows, { 0, 0 });
  result.col_idx.push_back(0);
  for (int j = 0; j < B.cols; j++) {
    for (int k = B.col_idx[j]; k < B.col_idx[j + 1]; k++) {
      col_not_zero = B.row[k];
      for (int i = A.col_idx[col_not_zero]; i < A.col_idx[col_not_zero + 1]; i++) {
        tmp[A.row[i]] += B.value[k] * A.value[i];
      }
    }
    for (int c = 0; c < A.rows; c++) {
      if (tmp[c] != std::complex<int>(0, 0)) {
        result.row.push_back(c);
        result.value.push_back(tmp[c]);
        tmp[c] = 0;
      }
    }
    result.col_idx.push_back(result.value.size());
  }
  return result;
}
