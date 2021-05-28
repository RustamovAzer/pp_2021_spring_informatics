// Copyright 2021 Tarasov Nikita
#ifndef MODULES_TASK_1_TARASOV_N_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
#define MODULES_TASK_1_TARASOV_N_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
#include <iostream>
#include <ctime>
#include <random>
#include <limits>
#include <vector>

std::vector<double> get_random_img(int _width, int _height);
void print_img(std::vector<double> pic, int width, int height);
std::vector<double> gaussian_core(double sigma);
std::vector<double> gaussian_filter(std::vector<double> img, int width, int height, double sigma);

#endif  // MODULES_TASK_1_TARASOV_N_GAUSSIAN_FILTER_GAUSSIAN_FILTER_H_
