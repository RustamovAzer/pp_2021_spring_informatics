//  Copyright 2021 Dobrov Pavel
#include "../../../modules/task_2/dobrov_p_m_Cannon/m_cannon.h"
#include <omp.h>
#include <vector>
#include <random>

std::vector<double> RandomMatrix(int n) {
    if (n <= 0)
        throw std::invalid_argument("Negative size");

    std::mt19937 generator;
    std::random_device device;
    generator.seed(device());
    std::uniform_real_distribution<double> distribution(0, 100);
    std::vector<double> matrix(n * n);

    for (int i = 0; i < n * n; i++) {
        matrix[i] = distribution(generator);
    }

    return matrix;
}

std::vector<double> seqMulti(const std::vector<double> &A, const std::vector<double> &B, int n) {
    if (n <= 0) {
        throw std::invalid_argument("The values of the matrix dimensions must be greater than 0");
    }
    if (static_cast<size_t>(n * n) != A.size() || static_cast<size_t>(n * n) != B.size()) {
        throw std::invalid_argument("Incorrect values");
    }

    std::vector<double> C(n * n);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                C[j * n + i] += A[j * n + k] * B[k * n + i];
            }
        }
    }

    return C;
}

std::vector<double> parMulti(const std::vector<double> &A, const std::vector<double> &B, int n) {
    if (n <= 0) {
        throw std::invalid_argument("The values of the matrix dimensions must be greater than 0");
    }
    if (static_cast<size_t>(n * n) != A.size() || static_cast<size_t>(n * n) != B.size()) {
        throw std::invalid_argument("Incorrect values");
    }

    std::vector<double> C(n * n);

    int q = static_cast<int>(std::sqrt(omp_get_max_threads()));
    int blockSize = n / q + n % q;

#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        int jStart = (threadNum / q) * blockSize;
        int jEnd = std::min(jStart + blockSize, n);

        int iStart = (threadNum % q) * blockSize;
        int iEnd = std::min(iStart + blockSize, n);

        for (int j = jStart; j < jEnd; j++) {
            for (int i = iStart; i < iEnd; i++) {
                for (int k = 0; k < n; k++) {
                    C[j * n + i] += A[j * n + k] * B[k * n + i];
                }
            }
        }
    }

    return C;
}
