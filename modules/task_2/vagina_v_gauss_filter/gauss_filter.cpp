// Copyright 2021 Vagina Valeria
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "../../../modules/task_2/vagina_v_gauss_filter/gauss_filter.h"

bool operator==(const Pixel &a, const Pixel &b) {
    return (a.r == b.r) && (a.g == b.g) && (a.b == b.b);
}

std::vector<double> generateGaussianKernel(int radius) {
    const int size = 2 * radius + 1;
    std::vector<double> res(size * size);
    const double sigma = 2;
    double sum = 0;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            sum += res[(i + radius) * size + j + radius] =
            static_cast<double>(exp((-1) * (i * i + j * j) / (sigma * sigma)));
        }
    }
    for (int i = 0; i < size * size; ++i) {
        res[i] /= sum;
    }

    return res;
}

std::vector<Pixel> generateRandomImage(int rows, int cols) {
    std::mt19937 gen;
    gen.seed((unsigned)time(0));
    std::uniform_int_distribution<int> dis(0, 255);
    std::vector<Pixel> result(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        result[i] = {static_cast<uint8_t>(dis(gen)),
        static_cast<uint8_t>(dis(gen)), static_cast<uint8_t>(dis(gen))};
    }
    return result;
}

std::vector<Pixel> gaussFilterOMP(const std::vector<Pixel> &a,
                               int rows, int cols) {
    std::vector<Pixel> res(a);
    if (rows * cols != static_cast<int>(res.size())) {
        throw "Matrix radiusensions do not match";
    }

    int radius = 3;
    std::vector<double> kernel = generateGaussianKernel(radius);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double r = 0, g = 0, b = 0;
            int gaussianIndex = 0;
            for (int l = -radius; l <= radius; ++l) {
                for (int k = -radius; k <= radius; ++k) {
                    int idX = std::max(0, std::min(i + l, rows - 1));
                    int idY = std::max(0, std::min(j + k, cols - 1));
                    Pixel p = a[idX * cols + idY];
                    r += kernel[gaussianIndex] * p.r;
                    g += kernel[gaussianIndex] * p.g;
                    b += kernel[gaussianIndex] * p.b;
                    ++gaussianIndex;
                }
            }
            Pixel &p = res[i * cols + j];
            p.r = std::max(0, std::min(static_cast<int>(r), 255));
            p.g = std::max(0, std::min(static_cast<int>(g), 255));
            p.b = std::max(0, std::min(static_cast<int>(b), 255));
        }
    }
    return res;
}
