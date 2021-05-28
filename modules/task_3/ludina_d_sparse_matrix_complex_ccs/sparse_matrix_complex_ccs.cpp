// Copyright 2021 Ludina Daria
#include "../../../modules/task_3/ludina_d_sparse_matrix_complex_ccs/sparse_matrix_complex_ccs.h"

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
    for (int j = col_idx[i]; j < col_idx[i + 1]; j++) {
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
  SparseMatrix result(A.rows, B.cols);
  A = A.transpose();
  result.col_idx.push_back(0);
  for (int i = 0; i < B.cols; i++) {
    for (int j = 0; j < A.cols; j++) {
      std::complex<int> sum = { 0, 0 };
      for (int k = B.col_idx[i]; k < B.col_idx[i + 1]; k++) {
        for (int n = A.col_idx[j]; n < A.col_idx[j + 1]; n++) {
          if (B.row[k] == A.row[n]) {
            sum += B.value[k] * A.value[n];
            break;
          }
        }
      }
      if (sum != std::complex<int>(0, 0)) {
        result.value.push_back(sum);
        result.row.push_back(j);
      }
    }
    result.col_idx.push_back(static_cast<int>(result.value.size()));
  }
  return result;
}

SparseMatrix getRandomSparseMatrix(int rows, int cols, int notZero) {
  SparseMatrix result;
  std::vector<std::complex<int>> val;
  std::vector<int> col_i, r;
  col_i.push_back(0);
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < notZero; i++) {
      val.push_back(std::complex<int>(
        static_cast<int>(gen() % 100),
        static_cast<int>(gen() % 100)));
      r.push_back(gen() % cols);
    }
    col_i.push_back(static_cast<int>(r.size()));
  }
  result.value = val;
  result.row = r;
  result.col_idx = col_i;
  return result;
}

SparseMatrix MultiplicationOpenMP(SparseMatrix A, SparseMatrix B) {
  if (A.cols != B.rows)
    throw "Size col A not equal size row B";
  SparseMatrix res(A.rows, B.cols);
  std::vector<std::vector<std::complex<int>>> val(res.cols);
  std::vector<std::vector<int>> r(res.cols);
  A = A.transpose();
  res.col_idx.push_back(0);
  int i, j, k;

#pragma omp parallel
  {
    std::vector<int> tmp(res.cols);
#pragma omp for private(j, k) schedule(static)
    for (i = 0; i < res.cols; i++) {
      tmp.assign(res.cols, -1);
      for (j = B.col_idx[i]; j < B.col_idx[i + 1]; j++) {
        tmp[B.row[j]] = j;
      }
      for (j = 0; j < res.cols; j++) {
        std::complex<int> sum = { 0, 0 };
        for (k = A.col_idx[j]; k < A.col_idx[j + 1]; k++) {
          if (tmp[A.row[k]] != -1) {
            sum += B.value[tmp[A.row[k]]] * A.value[k];
          }
        }
        if (sum != std::complex<int>(0, 0)) {
          val[i].emplace_back(sum);
          r[i].emplace_back(j);
        }
      }
    }
  }
  for (int i = 0; i < res.cols; i++) {
    res.value.insert(res.value.end(), val[i].begin(), val[i].end());
    res.row.insert(res.row.end(), r[i].begin(), r[i].end());
    res.col_idx.insert(res.col_idx.end(), r[i].begin(), r[i].end());
  }
  return res;
}

SparseMatrix MultiplicationTBB(SparseMatrix A, SparseMatrix B) {
  if (A.cols != B.rows)
    throw "Size col A not equal size row B";
  SparseMatrix res(A.rows, B.cols);
  std::vector<std::vector<std::complex<int>>> val(res.cols);
  std::vector<std::vector<int>> res_row(res.cols);
  A = A.transpose();
  res.col_idx.push_back(0);
  tbb::task_scheduler_init init;
  int my_grainsize = 10;
  tbb::parallel_for(tbb::blocked_range<int>(0, B.cols, my_grainsize),
    [&](const tbb::blocked_range<int>& r) {
    std::vector<int> tmp(res.cols);
    for (int i = r.begin(); i < r.end(); i++) {
      tmp.assign(res.cols, -1);
      for (int j = B.col_idx[i]; j < B.col_idx[i + 1]; j++) {
        tmp[B.row[j]] = j;
      }
      for (int j = 0; j < res.cols; j++) {
        std::complex<int> sum = { 0, 0 };
        for (int k = A.col_idx[j]; k < A.col_idx[j + 1]; k++) {
          if (tmp[A.row[k]] != -1) {
            sum += B.value[tmp[A.row[k]]] * A.value[k];
          }
        }
        if (sum != std::complex<int>(0, 0)) {
          val[i].emplace_back(sum);
          res_row[i].emplace_back(j);
        }
      }
    }
  });
  init.terminate();
  for (int i = 0; i < res.cols; i++) {
    res.value.insert(res.value.end(), val[i].begin(), val[i].end());
    res.row.insert(res.row.end(), res_row[i].begin(), res_row[i].end());
    res.col_idx.insert(res.col_idx.end(), res_row[i].begin(), res_row[i].end());
  }
  return res;
}
