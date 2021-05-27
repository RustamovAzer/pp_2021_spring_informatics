// Copyright 2021 Rustamov Azer

#ifndef MODULES_TASK_4_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_
#define MODULES_TASK_4_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_


#include <vector>

using Matrix = std::vector<int>;

Matrix generate_random_image(int w, int h, int min_y = 30, int max_y = 192);

void data_distribution( const int data_size, std::vector<int>& starts,
                        std::vector<int>& sizes, int& num_threads,
                        int& count, int& rem);

Matrix make_histogram(const Matrix& image, int w, int h);

int get_min_y(const Matrix& histogram);

int get_max_y(const Matrix& histogram);

void part_get_min_max_y(const Matrix& image, int start, int size,
                        std::vector<unsigned char>* min_vec,
                        std::vector<unsigned char>* max_vec, int i);

void better_get_max_y_std(const Matrix& image, const int& h, const int& w);

Matrix stretch_histogram(const Matrix& histogtram, const int& min_y, const int& max_y);

Matrix increase_contrast(const Matrix& image, int w, int h, const int& min_y, const int& max_y);

void increase_contrast_part(const Matrix* image, Matrix* result, int start, int size, int min_y, int max_y);

Matrix increase_contrast_std(const Matrix& image, int w, int h, const int& min_y, const int& max_y);

Matrix histogram_sretch_algorithm(const Matrix& image, const int w, const int h);

Matrix histogram_sretch_algorithm_std(const Matrix& image, const int w, const int h);

#endif  // MODULES_TASK_4_RUSTAMOV_A_HISTOGRAM_STRETCH_HISTOGRAM_STRETCH_H_
