// Copyright 2021 Kirillov Konstantin
#include <gtest/gtest.h>
#include <omp.h>
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
    Matrix gausFilterSeq(rows, std::vector<double>(cols));
    Matrix gausFilterPar(rows, std::vector<double>(cols));
    Matrix img(rows, std::vector<double>(cols));
    Matrix kernelMat = genMatrixKernel(radius, alpha);
    img = getRandomMatrix(rows, cols);
    gausFilterSeq = SequentialGausFilter(img, rows, cols, radius, alpha);
    gausFilterPar = ParallelGausFilter(img, rows, cols, radius, alpha);
    printMatrix(kernelMat, 2 * radius + 1, 2 * radius + 1);
    printMatrix(img, rows, cols);
    printMatrix(gausFilterSeq, rows, cols);
    printMatrix(gausFilterPar, rows, cols);
    ASSERT_NO_THROW(getRandomMatrix(rows, cols));
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_Static_3x3) {
    int rows = 3;
    int cols = 3;

    double alpha = 1;
    int radius = 1;
    Matrix imgDef(rows, std::vector<double>(cols));
    Matrix gausFilterSeq(rows, std::vector<double>(cols));
    Matrix gausFilterPar(rows, std::vector<double>(cols));
    Matrix imgControl(rows, std::vector<double>(cols));
    Matrix kernelMat = genMatrixKernel(radius, alpha);

    imgDef[0][0] = 187;
    imgDef[0][1] = 234;
    imgDef[0][2] = 59;
    imgDef[1][0] = 126;
    imgDef[1][1] = 235;
    imgDef[1][2] = 215;
    imgDef[2][0] = 118;
    imgDef[2][1] = 131;
    imgDef[2][2] = 42;

    imgControl[0][0] = 192;
    imgControl[0][1] = 215;
    imgControl[0][2] = 125;
    imgControl[1][0] = 134;
    imgControl[1][1] = 178;
    imgControl[1][2] = 178;
    imgControl[2][0] = 117;
    imgControl[2][1] = 105;
    imgControl[2][2] = 78;

    gausFilterSeq = SequentialGausFilter(imgDef, rows, cols, radius, alpha);
    gausFilterPar = ParallelGausFilter(imgDef, rows, cols, radius, alpha);
    ASSERT_EQ(imgControl, gausFilterSeq);
    ASSERT_EQ(imgControl, gausFilterPar);
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_1000x500x60thr) {
    int rows = 1000;
    int cols = 500;
    double alpha = 7;
    int radius = 2;
    Matrix gausFilterSeq(rows, std::vector<double>(cols));
    Matrix gausFilterPar(rows, std::vector<double>(cols));
    Matrix img(rows, std::vector<double>(cols));
    Matrix kernelMat = genMatrixKernel(radius, alpha);
    img = getRandomMatrix(rows, cols);
    double start = omp_get_wtime();
    gausFilterSeq = SequentialGausFilter(img, rows, cols, radius, alpha);
    double end = omp_get_wtime();
    std::cout << "Sequential time: " << end - start << std::endl;
    start = omp_get_wtime();
    omp_set_num_threads(60);
    gausFilterPar = ParallelGausFilter(img, rows, cols, radius, alpha);
    end = omp_get_wtime();
    std::cout << "Parallel time: " << end - start << std::endl;
    ASSERT_EQ(gausFilterSeq, gausFilterPar);
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_300x300x30thr) {
    int rows = 300;
    int cols = 300;
    double alpha = 7;
    int radius = 2;
    Matrix gausFilterSeq(rows, std::vector<double>(cols));
    Matrix gausFilterPar(rows, std::vector<double>(cols));
    Matrix img(rows, std::vector<double>(cols));
    Matrix kernelMat = genMatrixKernel(radius, alpha);
    img = getRandomMatrix(rows, cols);
    double start = omp_get_wtime();
    gausFilterSeq = SequentialGausFilter(img, rows, cols, radius, alpha);
    double end = omp_get_wtime();
    std::cout << "Sequential time: " << end - start << std::endl;
    start = omp_get_wtime();
    omp_set_num_threads(30);
    gausFilterPar = ParallelGausFilter(img, rows, cols, radius, alpha);
    end = omp_get_wtime();
    std::cout << "Parallel time: " << end - start << std::endl;
    ASSERT_EQ(gausFilterSeq, gausFilterPar);
}

TEST(Sequential_Gaus_Filter_Test, Test_Matrix_1500x1000x20thr) {
    int rows = 1500;
    int cols = 1000;
    double alpha = 7;
    int radius = 2;
    Matrix gausFilterSeq(rows, std::vector<double>(cols));
    Matrix gausFilterPar(rows, std::vector<double>(cols));
    Matrix img(rows, std::vector<double>(cols));
    Matrix kernelMat = genMatrixKernel(radius, alpha);
    img = getRandomMatrix(rows, cols);
    double start = omp_get_wtime();
    gausFilterSeq = SequentialGausFilter(img, rows, cols, radius, alpha);
    double end = omp_get_wtime();
    std::cout << "Sequential time: " << end - start << std::endl;
    start = omp_get_wtime();
    omp_set_num_threads(20);
    gausFilterPar = ParallelGausFilter(img, rows, cols, radius, alpha);
    end = omp_get_wtime();
    std::cout << "Parallel time: " << end - start << std::endl;
    ASSERT_EQ(gausFilterSeq, gausFilterPar);
}
