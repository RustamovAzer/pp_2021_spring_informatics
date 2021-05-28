// Copyright 2021 Aleksandrychev Andrey
#ifndef MODULES_TASK_2_ALEKSANDRYCHEV_A_SOBEL_OP_SOBEL_OP_H_
#define MODULES_TASK_2_ALEKSANDRYCHEV_A_SOBEL_OP_SOBEL_OP_H_
#include<vector>
#include<iostream>
#include<ctime>
#include<random>
#include<algorithm>

using Matrix = std::vector<std::vector<double>>;
Matrix create_binary_image(int rows, int cols);
Matrix sobel_op(const Matrix& image, int rows, int cols);
void printMatrix(Matrix img, int rows, int cols);

#endif  // MODULES_TASK_2_ALEKSANDRYCHEV_A_SOBEL_OP_SOBEL_OP_H_
