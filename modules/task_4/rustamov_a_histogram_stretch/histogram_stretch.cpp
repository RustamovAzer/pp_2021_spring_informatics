// Copyright 2021 Rustamov Azer

#include <random>
#include <iostream>
#include <algorithm>
#include "../../../modules/task_4/rustamov_a_histogram_stretch/histogram_stretch.h"
#include "../../../3rdparty/unapproved/unapproved.h"

Matrix generate_random_image(int w, int h, int min_y, int max_y) {
    if ((w <= 0) || (h <= 0) || (min_y < 0) || (max_y < 0) ||
        (min_y > 255) || (max_y > 255)) {
        throw std::runtime_error
            ("Incorrect input for 'generate_random_image'");
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

void data_distribution(const int data_size, std::vector<int>* limits,
                        int* num_threads, int* count) {
        *num_threads = std::thread::hardware_concurrency();
        if (*num_threads > data_size) {
            *num_threads = data_size;
        }
        *count = data_size / (*num_threads);
        (*limits).reserve((*num_threads) + 1);
        for (int i = 0; i < (*num_threads); i++)
            limits->at(i) = i * (*count);
        limits->at(*num_threads) = data_size;
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

void get_min_max_y_std(const Matrix& image, const int& h, const int& w,
                        int* min_y, int* max_y) {
    int size = h * w;
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > size) {
            num_threads = size;
    }
    int count = size / num_threads;
    std::vector<int> limits(num_threads + 1);
    for (int i = 0; i < num_threads; i++)
        limits[i] = i * count;
    limits[num_threads] = size;
    std::vector<unsigned char> min_vec(num_threads, 255);
    std::vector<unsigned char> max_vec(num_threads, 0);

    auto part_min_max_y = [&image, &min_vec, &max_vec, &limits]
        (int n) -> void {
        unsigned char local_min = 255;
        unsigned char local_max = 0;
        for (int i = limits[n]; i < limits[n + 1]; i++) {
            if (image[i] < local_min) {
                local_min = image[i];
            }
            if (image[i] > local_max) {
                local_max = image[i];
            }
        }
        min_vec[n] = local_min;
        max_vec[n] = local_max;
    };

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(part_min_max_y, i));
    }
    for (int i = 0; i < num_threads; i++)
        threads[i].join();
    *min_y = *std::min_element(min_vec.begin(), min_vec.end());
    *max_y = *std::max_element(max_vec.begin(), max_vec.end());
}

Matrix stretch_histogram(const Matrix& histogram,
                            const int& min_y, const int& max_y) {
    if (min_y >= max_y)
        throw std::runtime_error
            ("Cannot stretch histohram with provided min_y and max_y");
    Matrix result_histogram(256);

    for (int i = min_y; i < max_y + 1; i++) {
        result_histogram[
            round(static_cast<double>(255 * (i - min_y))
                / static_cast<double>(max_y - min_y))] = histogram[i];
    }
    return result_histogram;
}

Matrix increase_contrast(const Matrix& image, int w, int h,
                            const int& min_y, const int& max_y) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'increase_contrast'");
    if (min_y >= max_y)
        throw std::runtime_error
            ("Cannot stretch histohram with provided min_y and max_y");
    Matrix result_image(h * w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            result_image[i * w + j] =
                round(static_cast<double>(255 * (image[i * w + j] - min_y))
                    / static_cast<double>(max_y - min_y));
        }
    }
    return result_image;
}

Matrix increase_contrast_std(const Matrix& image, int w, int h,
                                const int& min_y, const int& max_y) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'increase_contrast'");
    if (min_y >= max_y)
        throw std::runtime_error
            ("Cannot stretch histohram with provided min_y and max_y");
    int size = h * w;
    Matrix result_image(size);
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > size) {
            num_threads = size;
    }
    int count = size / num_threads;
    std::vector<int> limits(num_threads + 1);
    for (int i = 0; i < num_threads; i++)
        limits[i] = i * count;
    limits[num_threads] = size;

    auto part_increase_contrast = [&image, &result_image,
                                    &limits, &min_y, &max_y]
        (int n) -> void {
        for (int i = limits[n]; i < limits[n + 1]; i++) {
            result_image[i] = round(static_cast<double>(255 * (image[i] - min_y))
                                    / static_cast<double>(max_y - min_y));
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(part_increase_contrast, i));
    }
    for (int i = 0; i < num_threads; i++)
        threads[i].join();

    return result_image;
}

Matrix histogram_sretch_algorithm(const Matrix& image,
                                    const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram'");
    Matrix histogram = make_histogram(image, w, h);
    int min_y, max_y;
    min_y = get_min_y(histogram);
    max_y = get_max_y(histogram);
    return increase_contrast(image, w, h, min_y, max_y);
}

Matrix histogram_sretch_algorithm_std(const Matrix& image,
                                        const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'make_histogram'");
    int min_y, max_y;
    get_min_max_y_std(image, h, w, &min_y, &max_y);
    return increase_contrast_std(image, w, h, min_y, max_y);
}
