// Copyright 2021 Romanyuk Sergey
#include <omp.h>
#include <vector>
#include <random>
#include <ctime>
#include "../../../modules/task_2/romanyuk_s_alg_cannon/cannon.h"

std::vector<double> genMatrix(int n) {
    if (n <= 0) {
        throw("Incorrect size!");
    }

    std::vector<double> matrix(n * n);

    std::mt19937 random;
    random.seed(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = random() % 100;
        }
    }
    return matrix;
}

std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A,
    const std::vector<double>& B, int n) {

    if (A.size() != B.size()) {
        throw "Matrices sizes differ";
    }

    std::vector<double> res(n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                res[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    return res;
}

std::vector<double> CannonMultiplication(std::vector<double> A,
    std::vector<double> B, int BlockSize) {

    if (A.size() != B.size()) {
        throw "Size of matrix different";
    }

    if (BlockSize <= 0) {
        throw "Size of Block must be > 0";
    }

    if (A.size() < (unsigned int)(BlockSize * BlockSize)) {
        throw "Block size cannot be larger than size of original matrices";
    }

    int size = static_cast<int>(sqrt(A.size()));
    std::vector<double> C(size * size);
    int BlockCount = size / BlockSize;

    for (int i = 0; i < size * size; i++) {
        C[i] = 0;
    }

    for (int i = 0; i < size; i += BlockCount) {
        for (int j = 0; j < size; j += BlockCount) {
            for (int k = 0; k < size; k += BlockCount) {
                for (int ii = i; ii < std::min(size, i + BlockCount); ii++) {
                    for (int jj = j;
                        jj < std::min(size, j + BlockCount); jj++) {
                        for (int kk = k;
                            kk < std::min(size, k + BlockCount); kk++) {
                            C[ii * size + jj] += A[ii * size + kk]
                                * B[kk * size + jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}



std::vector<double> parallelCannonMult(const std::vector<double>& A,
    const std::vector<double>& B, int Size) {
    if (A.size() != B.size()) {
        throw "Size of matrix different";
    }

    if (Size <= 0) {
        throw "Size of Block must be > 0";
    }

    if (A.size() < (unsigned int)(Size * Size)) {
        throw "Block size cannot be larger than size of original matrices";
    }

    int stage;
    int size = static_cast<int>(sqrt(A.size()));
    int cols = static_cast<int>(sqrt(Size));
    std::vector<double> res(size * size);
    int BlockSize = size / cols;
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int i1 = thread_num / static_cast<int>(sqrt(Size));
        int j1 = thread_num % static_cast<int>(sqrt(Size));
        for (stage = 0; stage < cols; stage++) {
            for (int i = i1 * BlockSize; i < (i1 + 1) * BlockSize; i++) {
                for (int j = j1 * BlockSize; j < (j1 + 1) * BlockSize; j++) {
                    for (int k = stage * BlockSize;
                        k < (stage + 1) * BlockSize; k++) {
                        res[i * size + j] += A[i * size + k] *
                            B[k * size + j];
                    }
                }
            }
        }
    }
    return res;
}

bool Difference(const double &x, const double &y) {
    return std::fabs(x - y) < 0.001;
}

bool DifferenceMatrix(const std::vector<double>&A,
    const std::vector<double>&B) {
    if (A.size() != B.size())
        throw "Different size of matrix";
    int size = A.size();
    for (int i = 0; i < size; i++)
        if (!Difference(A[i], B[i]))
            return false;
    return true;
}
