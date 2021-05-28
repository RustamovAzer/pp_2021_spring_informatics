// Copyright 2021 Pronkin Dmitry
#include "../../../modules/task_2/pronkin_d_gaussian_filter/gaussian_filter.h"

std::vector<double> getRandomImage(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("Incorrect image size!");
    }
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> image(height * width);
    for (int i = 0; i < height * width; i++) {
        image[i] = gen() / static_cast<double>(std::numeric_limits<int>::max());
    }
    return image;
}

std::vector<double> getGaussianKernel(double sigma) {
    double norm = 0.0;
    std::vector<double> KERNEL(KERNEL_SIZE);
    for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
        KERNEL[n + 1] = std::exp(-((n * n) / (2.0 * sigma * sigma))) / (std::sqrt(2.0 * M_PI) * sigma);
        norm += KERNEL[n + 1];
    }
    for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
        KERNEL[n + 1] /= norm;
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
    std::vector<double> horizontalKernelResult(height * width);
    std::vector<double> result(height * width);
    for (int i = width + 1; i < width * (height - 1) - 1; i++) {
        if (i % width == 0 || i % width == width - 1) continue;
        for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
            horizontalKernelResult[i] += image[i + n] * KERNEL[n + KERNEL_SIZE / 2];
        }
    }
    for (int i = width + 1; i < width * (height - 1) - 1; i++) {
        if (i % width == 0 || i % width == width - 1) continue;
        for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
            result[i] += horizontalKernelResult[i + n * width] * KERNEL[n + KERNEL_SIZE / 2];
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
    std::vector<double> horizontalKernelResult(height * width);
    std::vector<double> result(height * width);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = width + 1; i < width * (height - 1) - 1; i++) {
            if (i % width == 0 || i % width == width - 1) continue;
            for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
                horizontalKernelResult[i] += image[i + n] * KERNEL[n + KERNEL_SIZE / 2];
            }
        }
        #pragma omp for schedule(static)
        for (int i = width + 1; i < width * (height - 1) - 1; i++) {
            if (i % width == 0 || i % width == width - 1) continue;
            for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
                result[i] += horizontalKernelResult[i + n * width] * KERNEL[n + KERNEL_SIZE / 2];
            }
        }
    }
    return result;
}
