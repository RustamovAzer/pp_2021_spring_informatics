// Copyright 2021 Tarasov Nikita
#include "../../../modules/task_1/tarasov_n_gaussian_filter/gaussian_filter.h"

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

std::vector<double> gaussian_filter(std::vector<double> img, int width, int height, double sigma) {
    if (height * width != static_cast<int>(img.size())) throw "Error";
    if (height * width < 9) throw "Error";
    std::vector<double> core = gaussian_core(sigma);
    std::vector<double> result_img = img;

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            double res = 0;
            for (int n = -1; n <= 1; n++) {
                for (int z = -1; z <= 1; z++) {
                    res += img[(i + n) * height + j + z] * core[(n + 1) * 3 + (z + 1)];
                }
            }
            res = round(res * 1000) / 1000;
            result_img[width * i + j] = res;
        }
    }
    return result_img;
}
