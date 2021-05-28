// Copyright 2021 Pronkin Dmitry
#include "../../../modules/task_1/pronkin_d_gaussian_filter/gaussian_filter.h"

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
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
                horizontalKernelResult[y * width + x] += image[y * width + x + n] * KERNEL[n + KERNEL_SIZE / 2];
            }
        }
    }
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int n = -(KERNEL_SIZE / 2); n <= KERNEL_SIZE / 2; n++) {
                result[y * width + x] += horizontalKernelResult[(y + n) * width + x] * KERNEL[n + KERNEL_SIZE / 2];
            }
        }
    }
    return result;
}
