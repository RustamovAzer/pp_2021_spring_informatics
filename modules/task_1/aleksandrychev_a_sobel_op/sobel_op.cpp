// Copyright 2021 Aleksandrychev Andrey
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stack>
#include "../../../modules/task_1/aleksandrychev_a_sobel_op/sobel_op.h"

template <class T> T clamp(T tmp, int max, int min) {
    return (tmp > max) ? max : (tmp < min) ? min : tmp;
}

Matrix create_binary_image(int rows, int cols) {
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Fix size of Matrix");
    std::mt19937 gen;
    gen.seed(static_cast<int>(time(0)));
    Matrix def(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            def[i][j] = gen() % 255;
        }
    }
    return def;
}

void printMatrix(Matrix img, int rows, int cols) {
    std::cout << "Matrix" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << img[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix sobel_op(const Matrix& image, int rows, int cols) {
    Matrix res_image = image;

    Matrix Gx = {{-1, -2, -1}, {0,   0,  0}, {1,   2,  1}};

    Matrix Gy = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int pixel_x = 0;
            int pixel_y = 0;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    pixel_x += res_image[i + x - 1][j + y - 1]*Gx[x][y];
                    pixel_y += res_image[i + x - 1][j + y - 1]*Gy[x][y];
                }
            }
            res_image[i][j] = clamp(sqrt(pixel_x*pixel_x + pixel_y*pixel_y),
            255, 0);
        }
    }

    return res_image;
}
