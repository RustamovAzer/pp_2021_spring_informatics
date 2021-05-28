// Copyright 2021 Denis Ostapovich
#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <random>
#include "../../../modules/task_2/ostapovich_d_gaussian_blur/gaussian_blur.h"

std::vector<uint8_t> filter(const std::vector<uint8_t>& matrix, int width, int coreSize) {
    std::size_t length = matrix.size();

    if (length == 0 || width <= 0 || length % width != 0 || coreSize < 3 || coreSize % 2 == 0) {
        throw std::invalid_argument("Zero or negative argument passed");
    }

    int shift = coreSize / 2;
    auto core = calculateCore(coreSize);
    int height = length / width;
    std::vector<uint8_t> filtered(width * height);
    for (int i = shift; i < height - shift; i++) {
        for (int j = shift; j < width - shift; j++) {
            uint8_t result = 0;
            for (int m = -shift; m <= shift; m++) {
                for (int n = -shift; n <= shift; n++) {
                    result += matrix[width * (i + m) + j + n] * core[coreSize * (m + shift) + n + shift];
                }
            }
            filtered[width * i + j] = result;
        }
    }

    return filtered;
}

std::vector<uint8_t> filterParallel(const std::vector<uint8_t>& matrix, int width, int coreSize) {
    std::size_t length = matrix.size();

    if (length == 0 || width <= 0 || length % width != 0 || coreSize < 3 || coreSize % 2 == 0) {
        throw std::invalid_argument("Zero or negative argument passed");
    }

    int radius = coreSize / 2;
    std::vector<double> core = calculateCore(coreSize);
    int height = length / width;
    std::vector<uint8_t> filtered(width * height);

    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(static)

        for (int i = radius; i < height - radius; i++) {
            for (int j = radius; j < width - radius; j++) {
                uint8_t result = 0;
                for (int m = -radius; m <= radius; m++) {
                    for (int n = -radius; n <= radius; n++) {
                        result += matrix[width * (i + m) + j + n]
                            * core[coreSize * (m + radius) + n + radius];
                    }
                }
                filtered[width * i + j] = result;
            }
        }
    }

    return filtered;
}

std::vector<double> calculateCore(int size, double deviation) {
    if (size < 3) {
        throw std::out_of_range("Size must be greater or equal to 3");
    }
    if (size % 2 == 0) {
        throw std::invalid_argument("Size must be an odd number");
    }
    double sum = 0;
    double s = 2.0 * deviation * deviation;
    int shift = size / 2;
    std::vector<double> core(size * size);

    for (int i = -shift; i <= shift; i++) {
        for (int j = -shift; j <= shift; j++) {
            double res = (std::exp(-(i * i + j * j) / s)) / (M_PI * s);
            core[size * (i + shift) + j + shift] = res;
            sum += res;
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            core[size * i + j] /= sum;
        }
    }

    return core;
}

std::vector<uint8_t> getRandomMatrix(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Zero or negative arguments passed");
    }

    const int size = width * height;
    std::random_device device;
    std::mt19937 gen(device());
    std::vector<uint8_t> matrix(size);
    for (int i = 0; i < size; i++) {
        matrix[i] = gen() % 100;
    }

    return matrix;
}
