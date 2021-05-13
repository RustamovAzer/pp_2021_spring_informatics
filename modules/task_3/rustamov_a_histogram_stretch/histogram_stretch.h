// Copyright 2021 Rustamov Azer

#ifndef MODULES_TASK_3_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_
#define MODULES_TASK_3_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>


#include <vector>

using Matrix = std::vector<int>;

Matrix generate_random_image(int w, int h, int min_y = 30, int max_y = 192);
Matrix generate_random_image_tbb(int w, int h, int min_y = 30, int max_y = 192);

Matrix make_histogram(const Matrix& image, int w, int h);
Matrix make_histogram_tbb(const Matrix& image, int w, int h);
int get_min_y(const Matrix& histogram);
int get_max_y(const Matrix& histogram);
int better_get_max_y_tbb(const Matrix& image, const int& h, const int& w);
int better_get_min_y_tbb(const Matrix& image, const int& h, const int& w);
Matrix stretch_histogram(const Matrix& histogtram, const int& min_y, const int& max_y);
Matrix increase_contrast(const Matrix& image, int w, int h, const int& min_y, const int& max_y);
Matrix increase_contrast_tbb(const Matrix& image, int w, int h, const int& min_y, const int& max_y);

Matrix histogram_sretch_algorithm(const Matrix& image, const int w, const int h);
Matrix histogram_sretch_algorithm_tbb(const Matrix& image, const int w, const int h);

#endif  // MODULES_TASK_3_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_
