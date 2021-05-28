// Copyright 2021 Alibekov Murad
#ifndef MODULES_TASK_2_ALIBEKOV_M_CCS_COMPLEX_MATRIX_CCS_COMPLEX_MATRIX_H_
#define MODULES_TASK_2_ALIBEKOV_M_CCS_COMPLEX_MATRIX_CCS_COMPLEX_MATRIX_H_

#include <omp.h>

#include <vector>
#include <complex>
#include <iostream>
#include <utility>
#include <random>

const double ZERO_IN_CCS = 0.00001;

struct ccs_complex_matrix {
    int N;       // size of matrix (N x N)
    int NZ;      // count of non-zero elements

    // array of values (size = NZ):
    std::vector<std::complex<double> > values;

    // array of rows' numbers (size = NZ):
    std::vector<int> rows;

    // array of columns' indexes (size = N + 1):
    std::vector<int> col_indexes;

    ccs_complex_matrix(int _N, int _NZ) {
        N = _N;
        NZ = _NZ;
        values = std::vector<std::complex<double> >(NZ);
        rows = std::vector<int>(NZ);
        col_indexes = std::vector<int>(N + 1);
    }

    friend bool operator==(const ccs_complex_matrix &A, const ccs_complex_matrix &B);
};

ccs_complex_matrix generate_regular_ccs(int seed, int N, int count_in_col);

ccs_complex_matrix transpose(const ccs_complex_matrix &A);
ccs_complex_matrix naive_multiplicate(const ccs_complex_matrix &A, const ccs_complex_matrix &B);
ccs_complex_matrix optim_multiplicate(const ccs_complex_matrix &A, const ccs_complex_matrix &B);

ccs_complex_matrix naive_multiplicate_omp(const ccs_complex_matrix &A, const ccs_complex_matrix &B);
ccs_complex_matrix optim_multiplicate_omp(const ccs_complex_matrix &A, const ccs_complex_matrix &B);

void PrintCCSMatrix(const ccs_complex_matrix &A, bool isComplex = true);
void PrintDensificationOfCCSMatrix(const ccs_complex_matrix &A, bool isComplex = true);

bool operator==(const ccs_complex_matrix &A, const ccs_complex_matrix &B);

#endif  // MODULES_TASK_2_ALIBEKOV_M_CCS_COMPLEX_MATRIX_CCS_COMPLEX_MATRIX_H_
