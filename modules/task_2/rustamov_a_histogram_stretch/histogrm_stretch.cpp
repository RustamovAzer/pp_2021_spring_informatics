// Copyright 2021 Rustamov Azer

#include <random>
#include <iostream>
#include "../../../modules/task_1/rustamov_a_histogram_stretch/histogram_stretch.h"



Matrix generate_random_image(int w, int h, int min_y, int max_y) {
    if ((w <= 0) || (h <= 0) || (min_y < 0) || (max_y < 0) ||
        (min_y > 255) || (max_y > 255)) {
        throw std::runtime_error("Incorrect input for 'generate_random_image'");
    }
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_int_distribution<> rnd(min_y, max_y);
    Matrix image(h * w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            image[i * w + j] = static_cast<int>(rnd(mersenne));
        }
    }
    return image;
}

Matrix generate_random_image_parallel(int w, int h, int min_y, int max_y) {
    if ((w <= 0) || (h <= 0) || (min_y < 0) || (max_y < 0) ||
        (min_y > 255) || (max_y > 255)) {
        throw std::runtime_error("Incorrect input for 'generate_random_image'");
    }
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_int_distribution<> rnd(min_y, max_y);
    int size = h * w;
    Matrix image(size);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        image[i] = static_cast<int>(rnd(mersenne));
    }
    return image;
}

Matrix make_histogram(const Matrix& image, int w, int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram'");
    Matrix histogram(256, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            histogram[image[i * w + j]]++;
        }
    }
    return histogram;
}

Matrix make_histogram_parallel(const Matrix& image, int w, int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram_parallel'");
    Matrix histogram(256, 0);
    int size = w * h;
    #pragma omp parallel
    {
        Matrix hist_private(256, 0);

        #pragma omp for
        for (int i = 0; i < size; i++) {
            hist_private[image[i]]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += hist_private[i];
            }
        }
    }
    return histogram;
}

int get_min_y(const Matrix& histogram) {
    int min_y = -1;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] != 0) {
            min_y = i;
            break;
        }
    }
    return min_y;
}

int get_max_y(const Matrix& histogram) {
    int max_y = -1;
    for (int i = 255; i >= 0; i--) {
        if (histogram[i] != 0) {
            max_y = i;
            break;
        }
    }
    return max_y;
}

Matrix stretch_histogram(const Matrix& histogram, const int& min_y, const int& max_y) {
    if (min_y >= max_y)
        throw std::runtime_error("Cannot stretch histohram with provided min_y and max_y");
    Matrix result_histogram(256);

    for (int i = min_y; i < max_y + 1; i++) {
        result_histogram[255 * (i - min_y) / (max_y - min_y)] = histogram[i];
    }
    return result_histogram;
}

Matrix increase_contrast(const Matrix& image, int w, int h, const int& min_y, const int& max_y) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'increase_contrast'");
    if (min_y >= max_y)
        throw std::runtime_error("Cannot stretch histohram with provided min_y and max_y");
    Matrix result_image(h * w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            result_image[i * w + j] = 255 * (image[i * w + j] - min_y) / (max_y - min_y);
        }
    }
    return result_image;
}

Matrix increase_contrast_parallel(const Matrix& image, int w, int h, const int& min_y, const int& max_y) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'increase_contrast'");
    if (min_y >= max_y)
        throw std::runtime_error("Cannot stretch histohram with provided min_y and max_y");
    int size = h * w;
    Matrix result_image(size);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result_image[i] = 255 * (image[i] - min_y) / (max_y - min_y);
    }
    return result_image;
}

Matrix histogram_sretch_algorithm(const Matrix& image, const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram'");
    Matrix histogram = make_histogram(image, w, h);
    int min_y, max_y;
    min_y = get_min_y(histogram);
    max_y = get_max_y(histogram);
    return increase_contrast(image, w, h, min_y, max_y);
}

Matrix histogram_sretch_algorithm_parallel(const Matrix& image, const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram'");
    Matrix histogram = make_histogram_parallel(image, w, h);
    int min_y, max_y;
    min_y = get_min_y(histogram);
    max_y = get_max_y(histogram);
    return increase_contrast_parallel(image, w, h, min_y, max_y);
}
