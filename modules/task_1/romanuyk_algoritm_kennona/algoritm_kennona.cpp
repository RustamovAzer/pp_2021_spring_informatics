// Copyright 2020 Romanuyk Sergey

#include <vector>
#include <random>
#include "../../../modules/task_1/romanuyk_algoritm_kennona/algoritm_kennona.h"

std::vector<double> genMatrix(int n) {
    int SIZE = n * n;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> urd(-50, 50);
    std::vector<double> arr(SIZE);
    for (int i = 0; i < SIZE; i++) {
        arr[i] = urd(gen);
    }
    return arr;
}

std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A,
    const std::vector<double>& B, int n) {

    if (A.size() !=B.size()) {
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

std::vector<double> KennonMultiplication(std::vector<double> A,
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
