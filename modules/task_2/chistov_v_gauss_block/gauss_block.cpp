// Copyright 2021 Chistov Vladimir

#define _USE_MATH_DEFINES
#include <omp.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "../../../modules/task_2/chistov_v_gauss_block/gauss_block.h"

std::vector<double> Gauss_Sequential(std::vector<double> image, int width,
    int height) {
    std::vector<double> calc((width + 2) * (height + 2));
    for (int x = 0; x < width + 2; x++) {
        for (int y = 0; y < height + 2; y++) {
            if ((x == 0) || (y == 0) || (x == width + 1) || (y == height + 1)) {
                calc[x * (height + 2) + y] = 0;
            } else {
                calc[x * (height + 2) + y] = image[(x - 1) * height + y - 1];
            }
        }
    }
    int count = 0;
    std::vector<double> res(width * height);
    std::vector<double> G_Core = Gauss_Core(3);
    for (int x = 1; x < width + 1; x++) {
        for (int y = 1; y < height + 1; y++) {
            double sum = 0;
            int gc = 0;
            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    sum = sum + calc[(x + i) * (height + 2) + y + j]
                        * G_Core[gc];
                    gc++;
                }
            }
            res[count] = sum;
            count++;
        }
    }
    return res;
}

std::vector<double> Gauss_Parallel(std::vector<double> image, int width,
    int height) {
    std::vector<double> calc((width + 2) * (height + 2));
    int count = 0;
    std::vector<double> res(width * height);
    std::vector<double> G_Core = Gauss_Core(3);
    for (int x = 0; x < width + 2; x++) {
        for (int y = 0; y < height + 2; y++) {
            if ((x == 0) || (y == 0) || (x == width + 1) || (y == height + 1)) {
                calc[x * (height + 2) + y] = 0;
            } else {
                calc[x * (height + 2) + y] = image[(x - 1) * height + y - 1];
            }
        }
    }
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(static)
        for (int x = 1; x < width + 1; x++) {
            for (int y = 1; y < height + 1; y++) {
                double sum = 0;
                int gc = 0;
                for (int i = -1; i < 2; i++) {
                    for (int j = -1; j < 2; j++) {
                        sum = sum + calc[(x + i) * (height + 2) + y + j]
                            * G_Core[gc];
                        gc++;
                    }
                }
                res[count] = sum;
                count++;
            }
        }
    }
    return res;
}

std::vector<double> GenRandVec(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = gen() % 256;
    }
    return vec;
}

std::vector<double> Gauss_Core(int size, double dev) {
    double sum = 0;
    double s = 2.0 * dev * dev;
    int rad = size / 2;
    std::vector<double> core(size * size);
    for (int i = -rad; i <= rad; i++) {
        for (int j = -rad; j <= rad; j++) {
            double res = (std::exp(-(i * i + j * j) / s)) / (M_PI * s);
            core[size * (i + rad) + j + rad] = res;
            sum += res;
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            core[size * i + j] /= sum;
        }
    }
    return core;
}
