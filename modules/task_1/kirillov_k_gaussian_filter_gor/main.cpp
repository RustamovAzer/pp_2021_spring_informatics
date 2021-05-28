// Copyright 2021 Kirillov Konstantin
#include <gtest/gtest.h>
#include <vector>
#include<iostream>
#include "./gaus_filter_gor.h"

TEST(Sequential_Gaus_Filter_Test, Test_Throw_Work) {
    int rows = 0;
    int cols = -2;
    ASSERT_ANY_THROW(getRandomMatrix(rows, cols));
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_3x3) {
    int rows = 3;
    int cols = 3;
    double alpha = 5;
    int radius = 1;
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
    Matrix ImgWithGaus(rows, std::vector<double>(cols));
    Matrix Img(rows, std::vector<double>(cols));
    Matrix KernelMat = genMatrixKernel(radius, alpha);
    Img = getRandomMatrix(rows, cols);
    ImgWithGaus = SequentialGausFilter(Img, rows, cols, radius, alpha);
    printMatrix(KernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(Img, rows, cols);
    printMatrix(ImgWithGaus, rows, cols);
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_Static_3x3) {
    int rows = 3;
    int cols = 3;
    double alpha = 1;
    int radius = 1;
    Matrix ImgDef(rows, std::vector<double>(cols));
    Matrix ImgWithGaus(rows, std::vector<double>(cols));
    Matrix ImgControl(rows, std::vector<double>(cols));
    Matrix KernelMat = genMatrixKernel(radius, alpha);
    ImgDef[0][0] = 187;
    ImgDef[0][1] = 234;
    ImgDef[0][2] = 59;
    ImgDef[1][0] = 126;
    ImgDef[1][1] = 235;
    ImgDef[1][2] = 215;
    ImgDef[2][0] = 118;
    ImgDef[2][1] = 131;
    ImgDef[2][2] = 42;

    ImgControl[0][0] = 182;
    ImgControl[0][1] = 229;
    ImgControl[0][2] = 55;
    ImgControl[1][0] = 121;
    ImgControl[1][1] = 229;
    ImgControl[1][2] = 211;
    ImgControl[2][0] = 115;
    ImgControl[2][1] = 123;
    ImgControl[2][2] = 37;

    ImgWithGaus = SequentialGausFilter(ImgDef, rows, cols, radius, alpha);
    printMatrix(KernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(ImgDef, rows, cols);
    printMatrix(ImgWithGaus, rows, cols);
    printMatrix(ImgControl, rows, cols);
    ASSERT_EQ(ImgControl, ImgWithGaus);
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_5x5) {
    int rows = 5;
    int cols = 5;
    double alpha = 7;
    int radius = 2;
    Matrix ImgWithGaus(rows, std::vector<double>(cols));
    Matrix Img(rows, std::vector<double>(cols));
    Matrix KernelMat = genMatrixKernel(radius, alpha);
    Img = getRandomMatrix(rows, cols);
    ImgWithGaus = SequentialGausFilter(Img, rows, cols, radius, alpha);
    printMatrix(KernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(Img, rows, cols);
    printMatrix(ImgWithGaus, rows, cols);
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_4x2) {
    int rows = 4;
    int cols = 2;
    double alpha = 10;
    int radius = 2;
    Matrix ImgWithGaus(rows, std::vector<double>(cols));
    Matrix Img(rows, std::vector<double>(cols));
    Matrix KernelMat = genMatrixKernel(radius, alpha);
    Img = getRandomMatrix(rows, cols);
    ImgWithGaus = SequentialGausFilter(Img, rows, cols, radius, alpha);
    printMatrix(KernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(Img, rows, cols);
    printMatrix(ImgWithGaus, rows, cols);
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_2x4) {
    int rows = 2;
    int cols = 4;
    double alpha = 3;
    int radius = 2;
    Matrix ImgWithGaus(rows, std::vector<double>(cols));
    Matrix Img(rows, std::vector<double>(cols));
    Matrix KernelMat = genMatrixKernel(radius, alpha);
    Img = getRandomMatrix(rows, cols);
    ImgWithGaus = SequentialGausFilter(Img, rows, cols, radius, alpha);
    printMatrix(KernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(Img, rows, cols);
    printMatrix(ImgWithGaus, rows, cols);
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
}
