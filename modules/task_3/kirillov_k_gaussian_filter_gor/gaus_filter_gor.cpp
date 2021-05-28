// Copyright 2021 Kirillov Konstantin
#include "../../../modules/task_3/kirillov_k_gaussian_filter_gor/gaus_filter_gor.h"
Matrix getRandomMatrix(int rows, int cols) {
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

template <class T>
T clamp(T tmp, int max, int min) {
    return (tmp > max) ? max : (tmp < min) ? min : tmp;
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

Matrix genMatrixKernel(int radius, double alpha) {
    const int size = 2 * radius + 1;
    double normSum = 0;
    Matrix vecKernel(size, std::vector<double>(size));
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            vecKernel[i + radius][j + radius] =
            exp(-(i * i + j * j) / (alpha * alpha));
            normSum += vecKernel[i + radius][j + radius];
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            vecKernel[i][j] /= normSum;
        }
    }
    return vecKernel;
}

Matrix ParallelGausFilter(Matrix img, int rows, int cols,
    int radius, double alpha) {
    Matrix resImg(rows, std::vector<double>(cols));
    Matrix vecKernel = genMatrixKernel(radius, alpha);

    tbb::parallel_for(0, rows, [&](int x){
        tbb::parallel_for(0, cols, [&](int y){
            int resValue = 0;
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int x_ = clamp(x + j, rows - 1, 0);
                    int y_ = clamp(y + j, cols - 1, 0);
                    double value = img[x_][y_];
                    resValue += value * vecKernel[i + radius][j + radius];
                }
            }
            resValue = clamp(resValue, 255, 0);
            resImg[x][y] = resValue;
        });
    });
    return resImg;
}

Matrix SequentialGausFilter(Matrix img, int rows, int cols,
    int radius, double alpha) {
    Matrix resImg(rows, std::vector<double>(cols));
    Matrix vecKernel = genMatrixKernel(radius, alpha);
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            int resValue = 0;
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int x_ = clamp(x + j, rows - 1, 0);
                    int y_ = clamp(y + j, cols - 1, 0);
                    double value = img[x_][y_];
                    resValue += value * vecKernel[i + radius][j + radius];
                }
            }
            resValue = clamp(resValue, 255, 0);
            resImg[x][y] = resValue;
        }
    }
    return resImg;
}
