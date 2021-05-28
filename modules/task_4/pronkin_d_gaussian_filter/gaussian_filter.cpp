// Copyright 2021 Pronkin Dmitry

#define _USE_MATH_DEFINES

#include <vector>
#include <stdexcept>
#include <random>
#include <ctime>
#include <cmath>
#include <climits>
#include "../../../modules/task_4/pronkin_d_gaussian_filter/gaussian_filter.h"
#include "../../../3rdparty/unapproved/unapproved.h"

std::vector<double> getRandomImage(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("Incorrect image size!");
    }
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> image(height * width);
    for (int i = 0; i < height * width; i++) {
        image[i] = gen() / static_cast<double>(UINT_MAX);
    }
    return image;
}

std::vector<double> getGaussianKernel(double sigma) {
    double norm = 0.0;
    double s = 2 * sigma * sigma;
    std::vector<double> KERNEL(9);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            KERNEL[(y + 1) * 3 + x + 1] = std::exp(-(x * x + y * y) / (s)) / (M_PI * s);
            norm += KERNEL[(y + 1) * 3 + x + 1];
        }
    }
    for (int i = 0; i < 9; i++) {
        KERNEL[i] /= norm;
    }
    return KERNEL;
}

std::vector<double> seqGaussianFilter(const std::vector<double>& image, int height, int width, double sigma) {
    if (image.empty()) {
        throw std::invalid_argument("Image is empty!");
    }
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("Incorrect image size!");
    }
    if (image.size() != static_cast<size_t>(height * width)) {
        throw std::invalid_argument("Height and width options do not match image dimensions");
    }
    std::vector<double> KERNEL = getGaussianKernel(sigma);
    std::vector<double> result(height * width);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    result[y * width + x] += image[(y + j) * width + (x + i)] * KERNEL[(j + 1) * 3 + (i + 1)];
                }
            }
        }
    }
    return result;
}

std::vector<double> parGaussianFilter(const std::vector<double>& image, int height, int width, double sigma) {
    if (image.empty()) {
        throw std::invalid_argument("Image is empty!");
    }
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("Incorrect image size!");
    }
    if (image.size() != static_cast<size_t>(height * width)) {
        throw std::invalid_argument("Height and width options do not match image dimensions");
    }
    std::vector<double> KERNEL = getGaussianKernel(sigma);
    std::vector<double> result(height * width);

    const int nthreads = std::thread::hardware_concurrency();
    std::thread* threads = new std::thread[nthreads];

    int q = static_cast<int>(std::sqrt(nthreads));
    int blockSizeHeight = height / q + height % q;
    int blockSizeWidth = width / q + width % q;

    for (int i = 0; i < nthreads; i++) {
        threads[i] = std::thread([&](int thread) {
            int yStart = (thread / q) * blockSizeHeight + 1;
            int yEnd = std::min(yStart + blockSizeHeight, height - 1);

            int xStart = (thread % q) * blockSizeWidth + 1;
            int xEnd = std::min(xStart + blockSizeWidth, width - 1);

            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    double pixel = 0.0;
                    for (int j = -1; j <= 1; j++) {
                        for (int i = -1; i <= 1; i++) {
                            pixel += image[(y + j) * width + (x + i)] * KERNEL[(j + 1) * 3 + (i + 1)];
                        }
                    }
                    result[y * width + x] = pixel;
                }
            }
        }, i);
    }

    for (int i = 0; i < nthreads; i++) {
        threads[i].join();
    }

    delete[] threads;

    return result;
}
