// Copyright 2021 Napylov Evgenii
#include <tbb/tbb.h>
#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>

#include "../../../modules/task_3/napylov_e_contrast/contrast.h"

const int num_threads = 8;

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

std::pair<unsigned char, unsigned char> minmax_tbb(const VecImage& image) {
    std::vector<unsigned char> min_vec(num_threads);
    std::fill(min_vec.begin(), min_vec.end(), 255);
    std::vector<unsigned char> max_vec(num_threads);
    std::fill(max_vec.begin(), max_vec.end(), 0);

    int sz = image.size();

    tbb::task_scheduler_init init(num_threads);

    tbb::parallel_for(tbb::blocked_range<int>(0, sz),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++) {
                int thr_id = tbb::task_arena::current_thread_index();
                if (image[i] > max_vec[thr_id]) {
                    max_vec[thr_id] = image[i];
                }
                if (image[i] < min_vec[thr_id]) {
                    min_vec[thr_id] = image[i];
                }
            }
        });

    unsigned char max_col = *std::max_element(max_vec.begin(), max_vec.end());
    unsigned char min_col = *std::min_element(min_vec.begin(), min_vec.end());
    return std::pair<unsigned char, unsigned char>(min_col, max_col);
}

VecImage add_contrast_tbb(VecImage image) {
    auto minmax = minmax_tbb(image);
    unsigned char min_col = minmax.first;
    unsigned char max_col = minmax.second;

    int sz = image.size();

    tbb::task_scheduler_init init(num_threads);

    tbb::parallel_for(tbb::blocked_range<int>(0, sz),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++) {
                image[i] = round((static_cast<double>((image[i] - min_col))
                    / static_cast<double>((max_col - min_col))) * 255);
            }
        });

    return image;
}
