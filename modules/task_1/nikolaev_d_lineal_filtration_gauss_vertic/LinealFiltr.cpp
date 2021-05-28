// Copyright 2021 Nikolaev Denis

#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "../../../modules/task_1/nikolaev_d_lineal_filtration_gauss_vertic/LinealFiltr.h"

int clamp(int n, int upper, int lower) {
    return n <= lower ? lower : n >= upper ? upper : n;
}

Matrix createrandmatrix(int i, int j) {
    if ((i <= 0) || (j <= 0))
        throw std::invalid_argument("Invalid size of Matrix");
    std::mt19937 gen;
    gen.seed(static_cast<int>(time(0)));
    Matrix def(i, std::vector<double>(j));
    for (int k = 0; k < i; k++) {
        for (int t = 0; t < j; t++) {
            def[k][t] = gen() % 255;
        }
    }
    return def;
}

std::vector<double> transposition(const std::vector<double>& matrix,
    int i, int j) {
    std::vector<double> result(i * j);
    for (int k = 0; k < i; k++) {
        for (int t = 0; t < j; t++) {
            result[k + t * i] = matrix[k * j + t];
        }
    }
    return result;
}

void printMatrix(Matrix img, int i, int j) {
    std::cout << "Matrix" << std::endl;
    for (int k = 0; k < i; k++) {
        for (int t = 0; t < j; t++) {
            std::cout << img[k][t] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix gengausskernel(int R, int beta) {
    const int size = 2 * R + 1;
    double normSum = 0;
    Matrix vecKernel(size, std::vector<double>(size));
    for (int k = -R; k <= R; k++) {
        for (int t = -R; t <= R; t++) {
            vecKernel[k + R][t + R] =
                exp(-(k * k + t * t) / (beta * beta));
            normSum += vecKernel[k + R][t + R];
        }
    }
    for (int k = 0; k < size; k++) {
        for (int t = 0; t < size; t++) {
            vecKernel[k][t] /= normSum;
        }
    }
    return vecKernel;
}

Matrix sequentialgaussfilt(Matrix Image, int i, int j,
    int R, double beta) {
    Matrix resImg(i, std::vector<double>(j));
    Matrix vecKernel = gengausskernel(R, beta);
    for (int x = 0; x < i; x++) {
        for (int y = 0; y < j; y++) {
            int resValue = 0;
            for (int k = -R; k <= R; k++) {
                for (int t = -R; t <= R; t++) {
                    double value = Image[x][y];
                    resValue += value * vecKernel[k + R][t + R];
                }
            }
            resValue = clamp(resValue, 255, 0);
            resImg[x][y] = resValue;
        }
    }
    return resImg;
}
