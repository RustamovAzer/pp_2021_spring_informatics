// Copyright 2021 Kuznetsov Nikita
#include <cmath>
#include <random>
#include <limits>
#include "../../../modules/task_1/kuznetsov_n_mult_sparse_mat/sparse_matrix.h"

sMatrix::sMatrix(const std::vector<double>& _val, const std::vector<int>& _c_ind,
  const std::vector<int>& _rows, int _non_zero, int _size) {
  if (_size < 0) throw "Negative size";
  non_zero = static_cast<int>(_val.size());
  size = _size;
  val = _val;
  c_ind = _c_ind;
  rows = _rows;
}

sMatrix::sMatrix(int _size, int _non_zero, unsigned int key) {
  if (_size < 0) throw "Negative size";
  size = _size;
  non_zero = _non_zero;
  val.resize(non_zero);
  rows.resize(non_zero);
  c_ind.resize(size + 1);
  std::mt19937 generate;
  generate.seed(key);
  std::uniform_real_distribution<double> dgen(0.05, 1.0);
  for (int i = 0; i < non_zero; ++i) {
    double tmp = dgen(generate);
    val[i] = tmp * generate();
    rows[i] = generate() % size;
    ++c_ind[generate() % size + 1];
  }
  for (int i = 0; i < size; ++i) {
    c_ind[i + 1] += c_ind[i];
  }
}

bool sMatrix::operator==(const sMatrix& mat) {
  if (non_zero != mat.non_zero || size != mat.size) {
    return false;
  }
  return val == mat.val && c_ind == mat.c_ind && rows == mat.rows;
}

sMatrix sMatrix::operator*(const sMatrix& mat) {
  std::vector<double> result_val;
  std::vector<int> result_c_ind(size + 1);
  std::vector<int> result_rows;
  sMatrix tr_mat = this->transpose();
  const double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      double tmp = 0.0;
      for (int ci = mat.c_ind[i]; ci < mat.c_ind[i + 1]; ++ci) {
        for (int tr_ci = tr_mat.c_ind[j]; tr_ci < tr_mat.c_ind[j + 1]; ++tr_ci) {
          if (mat.rows[ci] == tr_mat.rows[tr_ci]) {
            tmp += tr_mat.val[tr_ci] * mat.val[ci];
            break;
          }
        }
      }
      if (tmp < eps || tmp < -eps) {
        result_val.push_back(tmp);
        result_rows.push_back(j);
        ++result_c_ind[i + 1];
      }
    }
    result_c_ind[i + 1] += result_c_ind[i];
  }
  return sMatrix(result_val, result_c_ind, result_rows, static_cast<int>(result_rows.size()), size);
}

sMatrix sMatrix::transpose() {
  std::vector<double> tr_val(val.size());
  std::vector<int> tr_ci(c_ind.size());
  std::vector<int> tr_rows(rows.size());
  for (int i = 0; i < non_zero; ++i) {
    ++tr_ci[rows[i] + 1];
  }
  int sum = 0;
  for (int i = 0; i <= size; ++i) {
    int tmp = tr_ci[i];
    tr_ci[i] = sum;
    sum += tmp;
  }
  for (int i = 0; i < size; i++) {
    for (int j = c_ind[i]; j < c_ind[i + 1]; j++) {
      int tmp_ind = tr_ci[rows[j] + 1];
      ++tr_ci[rows[j] + 1];
      tr_val[tmp_ind] = val[j];
      tr_rows[tmp_ind] = i;
    }
  }
  return sMatrix(tr_val, tr_ci, tr_rows, non_zero, size);
}
