// Copyright 2021 Pronkin Dmitry
#include "../../../modules/task_3/pronkin_d_gaussian_filter/gaussian_filter.h"

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
    tbb::parallel_for(
        tbb::blocked_range2d<int, int>(1, height - 1, 1, width - 1),
        [&](tbb::blocked_range2d<int, int> r) {
            for (int y = r.rows().begin(); y != r.rows().end(); y++) {
                for (int x = r.cols().begin(); x != r.cols().end(); x++) {
                    double pixel = 0.0;
                    for (int j = -1; j <= 1; j++) {
                        for (int i = -1; i <= 1; i++) {
                            pixel += image[(y + j) * width + (x + i)] * KERNEL[(j + 1) * 3 + (i + 1)];
                        }
                    }
                    result[y * width + x] = pixel;
                }
            }
        });
    return result;
}
