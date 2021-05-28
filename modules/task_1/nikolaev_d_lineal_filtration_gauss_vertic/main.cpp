// Copyright 2020 Nikolaev Denis
#include <gtest/gtest.h>
#include <vector>
#include<iostream>
#include "./LinealFiltr.h"

TEST(sequential_gaus_filt_Test, invalid_values) {
    int i = 0;
    int j = -2;
    ASSERT_ANY_THROW(createrandmatrix(i, j));
}

TEST(sequential_gaus_filt_Test, matrix_1x1) {
    int i = 1;
    int j = 1;
    double beta = 5;
    int R = 1;
    ASSERT_NO_THROW(createrandmatrix(i, j));
    Matrix imagewGauss(i, std::vector<double>(j));
    Matrix image(i, std::vector<double>(j));
    Matrix gausskernel = gengausskernel(R, beta);
    image = createrandmatrix(i, j);
    imagewGauss = sequentialgaussfilt(image, i, j, R, beta);
    // printMatrix(gausskernel, 2 * R + 1, 2 * R + 1);
    printMatrix(image, i, j);
    printMatrix(imagewGauss, i, j);
    ASSERT_NO_THROW(createrandmatrix(i, j));
}

TEST(sequential_gaus_filt_Test, matrix_3x3) {
    int i = 3;
    int j = 3;
    double beta = 1;
    int R = 1;
    Matrix imagedef(i, std::vector<double>(j));
    Matrix imagewGauss(i, std::vector<double>(j));
    Matrix imagecont(i, std::vector<double>(j));
    Matrix gausskernel = gengausskernel(R, beta);
    imagedef[0][0] = 187;
    imagedef[0][1] = 234;
    imagedef[0][2] = 59;
    imagedef[1][0] = 126;
    imagedef[1][1] = 235;
    imagedef[1][2] = 215;
    imagedef[2][0] = 118;
    imagedef[2][1] = 131;
    imagedef[2][2] = 42;

    imagecont[0][0] = 182;
    imagecont[0][1] = 229;
    imagecont[0][2] = 55;
    imagecont[1][0] = 121;
    imagecont[1][1] = 229;
    imagecont[1][2] = 211;
    imagecont[2][0] = 115;
    imagecont[2][1] = 123;
    imagecont[2][2] = 37;

    imagewGauss = sequentialgaussfilt(imagedef, i, j, R, beta);
    printMatrix(gausskernel, 2 * R + 1, 2 * R + 1);
    printMatrix(imagedef, i, j);
    printMatrix(imagewGauss, i, j);
    printMatrix(imagecont, i, j);
    ASSERT_EQ(imagecont, imagewGauss);
}

TEST(sequential_gaus_filt_Test, matrix_5x5) {
    int i = 5;
    int j = 5;
    double beta = 7;
    int R = 2;
    Matrix imagewGauss(i, std::vector<double>(j));
    Matrix image(i, std::vector<double>(j));
    Matrix gausskernel = gengausskernel(R, beta);
    image = createrandmatrix(i, j);
    imagewGauss = sequentialgaussfilt(image, i, j, R, beta);
    // printMatrix(gausskernel, 2 * R + 1, 2 * R + 1);
    printMatrix(image, i, j);
    printMatrix(imagewGauss, i, j);
    ASSERT_NO_THROW(createrandmatrix(i, j));
}

TEST(sequential_gaus_filt_Test, matrix_7x7) {
    int i = 7;
    int j = 7;
    double beta = 7;
    int R = 2;
    Matrix imagewGauss(i, std::vector<double>(j));
    Matrix image(i, std::vector<double>(j));
    Matrix gausskernel = gengausskernel(R, beta);
    image = createrandmatrix(i, j);
    imagewGauss = sequentialgaussfilt(image, i, j, R, beta);
    printMatrix(gausskernel, 2 * R + 1, 2 * R + 1);
    printMatrix(image, i, j);
    printMatrix(imagewGauss, i, j);
    ASSERT_NO_THROW(createrandmatrix(i, j));
}
