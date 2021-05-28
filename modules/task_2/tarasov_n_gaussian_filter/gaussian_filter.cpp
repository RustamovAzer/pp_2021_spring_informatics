// Copyright 2021 Tarasov Nikita
#include "../../../modules/task_2/tarasov_n_gaussian_filter/gaussian_filter.h"

#define PI 3.1415926535

#define CLR 256

std::vector<double> get_random_img(int _width, int _height) {
    int _size_img = _width * _height;
    if (_size_img < 1) throw "Error";
    std::default_random_engine random;
    random.seed(static_cast<unsigned int>(std::time(0)));
    std::vector<double> pic(_size_img);
    for (int i = 0; i < _size_img; i++) {
        pic[i] = random() % CLR;
    }
    return pic;
}

void print_img(std::vector<double> pic, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            std::cout << pic[i * height + j] << " ";
        std::cout << std::endl;
    }
}

std::vector<double> gaussian_core(double sigma) {
    std::vector<double> result_core(9);
    double normalization = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            double numerator = std::exp(-((i * i + j * j) / (2 * sigma * sigma)));
            double denominator = (2 * PI * sigma * sigma);
            result_core[(i + 1) * 3 + (j + 1)] = numerator / denominator;
            normalization += result_core[(i + 1) * 3 + (j + 1)];
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result_core[i * 3 + j] /= normalization;
        }
    }
    return result_core;
}

std::vector<double> expansion_img(std::vector<double> img, int width, int height) {
    int exp_width = width + 2;
    int exp_height = height + 2;
        std::vector<double> expansion_img(exp_width * exp_height);
    for (int i = 0; i < width; i++) {
        expansion_img[i + 1] = img[i];
        expansion_img[exp_width * (exp_height - 1) + 1 + i] = img[height * (width - 1) + i];
    }

    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            expansion_img[(i + 1) * exp_height + j + 1] = img[i * height + j];
        }
    }

    for (int j = 0; j < exp_height; j++) {
        expansion_img[exp_width * j] = expansion_img[exp_width * j + 1];
        expansion_img[exp_width * (j + 1) - 1] = expansion_img[exp_width * (j + 1) - 2];
    }

    return expansion_img;
}

std::vector<double> gaussian_filter(std::vector<double> img, int width, int height, double sigma) {
    if (height * width != static_cast<int>(img.size())) throw "Error";
    if (height * width < 9) throw "Error";

    std::vector<double> core = gaussian_core(sigma);
    std::vector<double> result_img = img;

    int exp_width = width + 2;
    int exp_height = height + 2;

    std::vector<double> exp_img = expansion_img(img, width, height);

    for (int i = 1; i < exp_height - 1; i++) {
        for (int j = 1; j < exp_width - 1; j++) {
            double res = 0;
            for (int n = -1; n <= 1; n++) {
                for (int z = -1; z <= 1; z++) {
                    res += exp_img[(i + n) * exp_height + j + z] * core[(n + 1) * 3 + (z + 1)];
                }
            }
            res = round(res * 1000) / 1000;
            result_img[height * (i - 1) + (j - 1)] = res;
        }
    }

    return result_img;
}

std::vector<double> gaussian_filter_omp(std::vector<double> img, int width, int height, double sigma) {
    if (height * width != static_cast<int>(img.size())) throw "Error";
    if (height * width < 9) throw "Error";

    int nested_threads_num = 2;
    omp_set_num_threads(nested_threads_num);
    omp_set_nested(true);

    std::vector<double> core = gaussian_core(sigma);
    std::vector<double> result_img = img;

    int exp_width = width + 2;
    int exp_height = height + 2;

    std::vector<double> exp_img = expansion_img(img, width, height);

#pragma omp parallel for
    for (int i = 1; i < exp_height - 1; i++) {
#pragma omp parallel for
        for (int j = 1; j < exp_width - 1; j++) {
            double res = 0;
            for (int n = -1; n <= 1; n++) {
                for (int z = -1; z <= 1; z++) {
                    res += exp_img[(i + n) * exp_height + j + z] * core[(n + 1) * 3 + (z + 1)];
                }
            }
            res = round(res * 1000) / 1000;
            result_img[height * (i - 1) + (j - 1)] = res;
        }
    }

    return result_img;
}
