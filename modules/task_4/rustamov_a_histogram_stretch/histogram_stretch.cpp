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

void data_distribution(const int data_size, std::vector<int>* starts,
                        std::vector<int>* sizes, int* num_threads,
                        int* count, int* rem) {
        *num_threads = std::thread::hardware_concurrency();
        if (*num_threads > data_size) {
            *num_threads = data_size;
        }
        *count = data_size / *num_threads;
        *rem = data_size % *num_threads;
        int start = 0;
        for (int i = 0; i < *num_threads; i++) {
            int size;
            if (i != *num_threads - 1) {
                size = *count;
            } else {
                size = *count + *rem;
            }
            (*sizes).push_back(size);
            if (i != 0) {
                start += sizes->at(i - 1);
            }
            (*starts).push_back(start);
        }
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

void part_get_min_max_y(const Matrix& image, int start, int size,
                        std::vector<unsigned char>* min_y_local,
                        std::vector<unsigned char>* max_y_local, int n) {
    for (int i = 0; i < size; i++) {
        if (image[start + i] < min_y_local->at(n))
            min_y_local->at(n) = image[start + i];
        if (image[start + i] > max_y_local->at(n))
            max_y_local->at(n) = image[start + i];
    }
}

void get_min_max_y_std(const Matrix& image, const int& h, const int& w,
                        int* min_y, int* max_y) {
    int size = h * w;
    std::vector<int> starts;
    std::vector<int> sizes;
    int num_threads;
    int count;
    int rem;
    data_distribution(size, &starts, &sizes, &num_threads, &count, &rem);

    std::vector<unsigned char> min_y_local(num_threads, 255);
    std::vector<unsigned char> max_y_local(num_threads, 0);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(part_get_min_max_y,
                                        image, starts[i], sizes[i],
                                        &min_y_local, &max_y_local, i));
    }
    for (auto &t : threads)
        t.join();

    *min_y = *std::min_element(min_y_local.begin(), min_y_local.end());
    *max_y = *std::max_element(max_y_local.begin(), max_y_local.end());
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

void increase_contrast_part(const Matrix* image, Matrix* result,
                            int start, int size, int min_y, int max_y) {
    for (int i = 0; i < size; i++) {
        result->at(start + i) =
            round(static_cast<double>(255 * (image->at(start + i) - min_y))
                / static_cast<double>(max_y - min_y));
    }
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
    std::vector<int> starts;
    std::vector<int> sizes;
    int num_threads;
    int count;
    int rem;
    data_distribution(size, &starts, &sizes, &num_threads, &count, &rem);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        threads.push_back(
            std::thread(
                increase_contrast_part, &image, &result_image,
                starts[i], sizes[i], min_y, max_y));
    }
    for (int i = 0; i < num_threads; i++)
        threads[i].join();

    return result_image;
}

Matrix histogram_sretch_algorithm(const Matrix& image,
                                    const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'histogram_sretch_algorithm'");
    Matrix histogram = make_histogram(image, w, h);
    int min_y, max_y;
    min_y = get_min_y(histogram);
    max_y = get_max_y(histogram);
    return increase_contrast(image, w, h, min_y, max_y);
}

Matrix histogram_sretch_algorithm_std(const Matrix& image,
                                        const int w, const int h) {
    if ((w <= 0) || (h <= 0))
        throw std::runtime_error("Incorrect input for 'histogram_sretch_algorithm_std'");
    int min_y, max_y;
    get_min_max_y_std(image, h, w, &min_y, &max_y);
    return increase_contrast_std(image, w, h, min_y, max_y);
}
