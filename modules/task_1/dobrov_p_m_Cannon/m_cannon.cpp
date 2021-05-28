//  Copyright 2021 Dobrov Pavel
#include "../../../modules/task_1/dobrov_p_m_Cannon/m_cannon.h"
#include <vector>
#include <random>

std::vector<double> RandomMatrix(int n, int m) {
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Negative size");

    std::mt19937 generator;
    std::random_device device;
    generator.seed(device());
    std::uniform_real_distribution<double> distribution(0, 100);
    std::vector<double> matrix(n * m);

    for (int i = 0; i < n * m; i++) {
        matrix[i] = distribution(generator);
    }

    return matrix;
}

std::vector<double> MatrixMulti(const std::vector<double> &A, const std::vector<double> &B, int m, int n, int l) {
    if (m <= 0 || n <= 0 || l <= 0) {
        throw std::invalid_argument("The values of the matrix dimensions must be greater than 0");
    }
    if (static_cast<size_t>(m * n) != A.size() || static_cast<size_t>(n * l) != B.size()) {
        throw std::invalid_argument("Incorrect values");
    }

    std::vector<double> C(m * l);

    int threads = THREADS;
    int blockSize = std::min(m / static_cast<int>(std::sqrt(threads)), n / static_cast<int>(std::sqrt(threads)));
    blockSize == 0 ? blockSize = 1 : 0;

    for (int jj = 0; jj < m; jj += blockSize) {
       int jjMin = std::min(jj + blockSize, m);
       for (int kk = 0; kk < n; kk += blockSize) {
           int kkMin = std::min(kk + blockSize, n);
           for (int i = 0; i < m; i++) {
               for (int k = kk; k < kkMin; k++) {
                   for (int j = jj; j < jjMin; j++) {
                      C[i * l + j] += A[i * n + k] * B[k * l + j];
                   }
                }
            }
        }
    }

    return C;
}

std::vector<double> NaiveMulti(const std::vector<double> &A, const std::vector<double> &B, int m, int n, int l) {
    if (m <= 0 || n <= 0 || l <= 0) {
        throw std::invalid_argument("The values of the matrix dimensions must be greater than 0");
    }
    if (static_cast<size_t>(m * n) != A.size() || static_cast<size_t>(n * l) != B.size()) {
        throw std::invalid_argument("Incorrect values");
    }

    std::vector<double> C(m * l);

     for (int i = 0; i < m; i++)
        for (int j = 0; j < l; j++)
            for (int k = 0; k < n; k++)
                C[i * l + j] += A[i * n + k] * B[k * l + j];

    return C;
}
