// Copyright 2021 Kuznetsov Nikita
#include <omp.h>
#include <random>
#include <cmath>
#include <limits>
#include "../../../modules/task_2/kuznetsov_n_mult_sparse_mat_omp/sparse_matrix.h"

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

sMatrix::sMatrix(const std::vector<double>& _val,
  const std::vector<int>& _c_ind, const std::vector<int>& _rows,
  int _non_zero, int _size) {
  if (_size < 0) throw "Negative size";
  non_zero = static_cast<int>(_val.size());
  size = _size;
  val = _val;
  c_ind = _c_ind;
  rows = _rows;
}

sMatrix sMatrix::operator*(const sMatrix& mat) {
  std::vector<double> result_val;
  std::vector<int> result_ci(size + 1);
  std::vector<int> result_rows;
  sMatrix tr_mat = this->transpose();
  const double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      double tmp = 0.0;
      for (int ci = mat.c_ind[i]; ci < mat.c_ind[i + 1]; ++ci) {
        for (int tr_ci = tr_mat.c_ind[j]; tr_ci < tr_mat.c_ind[j + 1];
          ++tr_ci) {
          if (mat.rows[ci] == tr_mat.rows[tr_ci]) {
            tmp += tr_mat.val[tr_ci] * mat.val[ci];
            break;
          }
        }
      }
      if (tmp > eps || tmp < -eps) {
        result_val.push_back(tmp);
        result_rows.push_back(j);
        ++result_ci[i + 1];
      }
    }
    result_ci[i + 1] += result_ci[i];
  }
  return sMatrix(result_val, result_ci, result_rows, static_cast<int>(
    result_rows.size()), size);
}

bool sMatrix::operator==(const sMatrix& mat) {
  if (non_zero != mat.non_zero || size != mat.size) {
    return false;
  }
  return val == mat.val && c_ind == mat.c_ind && rows == mat.rows;
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

sMatrix sMatrix::pMult(const sMatrix& mat, int threads) {
  std::vector<double> result_val;
  std::vector<int> result_ci(size + 1);
  std::vector<int> result_rows;
  sMatrix transposeMat = this->transpose();

  std::vector<std::vector<std::vector<double>>> local(threads);
  const double eps = std::numeric_limits<double>::epsilon();

#pragma omp parallel num_threads(threads)
  {
    int threads_count = omp_get_thread_num();
    local[threads_count].resize(3);
    local[threads_count][2].resize(size + 1, 0);

#pragma omp for
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        double tmp = 0.0;
        for (int k = mat.c_ind[i]; k < mat.c_ind[i + 1]; ++k) {
          for (int l = transposeMat.c_ind[j]; l <
            transposeMat.c_ind[j + 1]; ++l) {
            if (mat.rows[k] == transposeMat.rows[l]) {
              tmp += mat.val[k] * transposeMat.val[l];
              break;
            }
          }
        }
        if (tmp > eps || tmp < -eps) {
          local[threads_count][0].push_back(tmp);
          local[threads_count][1].push_back(j);
          ++local[threads_count][2][i + 1] += local[threads_count][2][i];
        }
      }
      local[threads_count][2][i + 1] += local[threads_count][2][i];
    }
  }
  for (int i = 0; i < threads; i++) {
    result_val.insert(result_val.end(), local[i][0].begin(), local[i][0].end());
    result_rows.insert(result_rows.end(), local[i][1].begin(),
      local[i][1].end());
    result_ci.insert(result_ci.end(), local[i][2].begin(), local[i][2].end());
  }
  return sMatrix(result_val, result_ci, result_rows,
    static_cast<int>(result_rows.size()), size);
}
