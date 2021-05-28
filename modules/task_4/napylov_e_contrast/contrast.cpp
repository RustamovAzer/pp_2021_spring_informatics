// Copyright 2021 Napylov Evgenii

#include <iostream>
#include <cassert>
#include <random>
#include <ctime>
#include <algorithm>
#include "../../../3rdparty/unapproved/unapproved.h"  // thread

#include "../../../modules/task_4/napylov_e_contrast/contrast.h"

void print_vec(const VecImage& vec) {
    for (auto val : vec) {
        std::cout << static_cast<int>(val) << ' ';
    }
    std::cout << std::endl;
}

VecImage RandomVector(int size) {
    static std::mt19937 gen(time(0));
    VecImage result(size);
    std::uniform_int_distribution<unsigned int> distr(0, 255);
    for (int i = 0; i < size; i++) {
        result[i] = static_cast<unsigned char>(distr(gen));
    }
    return result;
}

VecImage add_contrast(VecImage image) {
    unsigned char min = *std::min_element(image.begin(), image.end());
    unsigned char max = *std::max_element(image.begin(), image.end());
    if (max == min) {
        return image;
    } else {
        for (size_t i = 0; i < image.size(); i++) {
            image[i] = round((static_cast<double>((image[i] - min))
                / static_cast<double>((max - min))) * 255);
        }
        return image;
    }
}

void minmax_tf(const VecImage& image, int start, int size,
    std::vector<unsigned char>* min_vec,
    std::vector<unsigned char>* max_vec, int num) {
    for (int i = start; i < start+size; i++) {
        if (image[i] > max_vec->at(num)) {
            max_vec->at(num) = image[i];
        }
        if (image[i] < min_vec->at(num)) {
            min_vec->at(num) = image[i];
        }
    }
}

void contrast_tf(VecImage* image, int start, int size, int min_c, int max_c) {
    for (int i = start; i < start+size; i++) {
        image->at(i) = round((static_cast<double>((image->at(i) - min_c))
                    / static_cast<double>((max_c - min_c))) * 255);
    }
}

VecImage add_contrast_std(VecImage image) {
    int num_threads = 8;
    std::vector<std::thread> threads_vec;

    // remove unnecessary threads if there are more than the number of pixels
    if (num_threads > static_cast<int>(image.size())) {
        num_threads = image.size();
    }

    const int count = image.size() / num_threads;
    const int rem = image.size() % num_threads;

    std::vector<unsigned char> min_vec(num_threads);
    std::fill(min_vec.begin(), min_vec.end(), 255);
    std::vector<unsigned char> max_vec(num_threads);
    std::fill(max_vec.begin(), max_vec.end(), 0);

    std::vector<int> starts;
    std::vector<int> sizes;

    int start = 0;
    // data distr. and min/max team
    for (int i = 0; i < num_threads; i++) {
        int size;
        if (i != num_threads - 1) {
            size = count;
        } else {
            size = count + rem;
        }
        sizes.push_back(size);
        if (i != 0) {
            start += sizes[i - 1];
        }
        starts.push_back(start);
        threads_vec.push_back(std::thread(minmax_tf, image, start, size,
                                &min_vec, &max_vec, i));
    }

    // blocks the main thread -> synchronized
    for (int i = 0; i < num_threads; i++) threads_vec[i].join();

    unsigned char min_c = *std::min_element(min_vec.begin(), min_vec.end());
    unsigned char max_c = *std::max_element(max_vec.begin(), max_vec.end());

    threads_vec.clear();

    // stretching team
    for (int i = 0; i < num_threads; i++) {
        threads_vec.push_back(std::thread(contrast_tf, &image,
                                    starts[i], sizes[i], min_c, max_c));
    }

    for (int i = 0; i < num_threads; i++) threads_vec[i].join();

    return image;
}
