// Copyright 2021 Denis Ostapovich
#include <gtest/gtest.h>
#include <vector>
// #include <chrono>
#include "./gaussian_blur.h"

TEST(STD_Gaussian_blur, GetRandomMatrix_Throws_On_Zero_Args) {
    const int width = 0;
    const int height = 0;

    ASSERT_ANY_THROW(getRandomMatrix(width, height));
}

TEST(STD_Gaussian_blur, GetRandomMatrix_Throws_On_Negative_Args) {
    const int width = -1;
    const int height = -1;

    ASSERT_ANY_THROW(getRandomMatrix(width, height));
}

TEST(STD_Gaussian_blur, GetRandomMatrix_No_Throw_On_Valid_Args) {
    const int width = 3;
    const int height = 5;

    ASSERT_NO_THROW(getRandomMatrix(width, height));
}

TEST(STD_Gaussian_blur, GetRandomMatrix_Returns_Vector_Of_Valid_Size) {
    const int width = 2;
    const int height = 7;

    auto vec = getRandomMatrix(width, height);

    ASSERT_EQ(vec.size(), (size_t)14);
}

TEST(STD_Gaussian_blur, CalculateCore_Returns_Valid_Core) {
    const int size = 5;

    auto core = calculateCore(size);

    for (double num : core) {
        ASSERT_NE(num, 0);
    }
}

TEST(STD_Gaussian_blur, Filter_Throws_On_Zero_Size_Matrix) {
    std::vector<uint8_t> mat;
    const int width = 10;

    ASSERT_ANY_THROW(filter(mat, width));
}

TEST(STD_Gaussian_blur, Filter_Throws_On_Zero_Or_Negative_Width_Value) {
    const int width = 11;
    const int height = 21;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filter(mat, 0));
    ASSERT_ANY_THROW(filter(mat, -1));
}

TEST(STD_Gaussian_blur, Filter_Throws_Invalid_Width_Value) {
    const int width = 11;
    const int height = 21;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filter(mat, 10));
}

TEST(STD_Gaussian_blur, Filter_Throws_On_Invalid_Core_Size) {
    const int width = 11;
    const int height = 21;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filter(mat, width, 0));
    ASSERT_ANY_THROW(filter(mat, width, -1));
    ASSERT_ANY_THROW(filter(mat, width, 4));
}

TEST(STD_Gaussian_blur, Filter_No_Throw_On_Valid_Args) {
    const int width = 21;
    const int height = 31;
    const int coreSize = 3;

    auto mat = getRandomMatrix(width, height);

    ASSERT_NO_THROW(filter(mat, width, coreSize));
}

TEST(STD_Gaussian_blur, FilterParallel_Throws_On_Zero_Size_Matrix) {
    std::vector<uint8_t> mat;
    const int width = 10;

    ASSERT_ANY_THROW(filterParallel(mat, width));
}

TEST(STD_Gaussian_blur, FilterParallel_Throws_On_Zero_Or_Negative_Width_Value) {
    const int width = 11;
    const int height = 21;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filterParallel(mat, 0));
    ASSERT_ANY_THROW(filterParallel(mat, -1));
}

TEST(STD_Gaussian_blur, FilterParallel_Throws_On_Invalid_Width_Value) {
    const int width = 10;
    const int height = 20;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filterParallel(mat, 3));
}

TEST(STD_Gaussian_blur, FilterParallel_Throws_On_Invalid_Core_Size) {
    const int width = 11;
    const int height = 21;

    auto mat = getRandomMatrix(width, height);

    ASSERT_ANY_THROW(filterParallel(mat, width, 0));
    ASSERT_ANY_THROW(filterParallel(mat, width, -5));
    ASSERT_ANY_THROW(filterParallel(mat, width, 8));
}

TEST(STD_Gaussian_blur, FilterParallel_No_Throw_On_Valid_Args) {
    const int width = 10;
    const int height = 20;

    auto mat = getRandomMatrix(width, height);

    ASSERT_NO_THROW(filterParallel(mat, width));
}

TEST(STD_Gaussian_blur, FilterParallel_Result_Equals_To_Filter_Result) {
    const int width = 5;
    const int height = 5;
    const int coreSize = 3;
    auto mat = getRandomMatrix(width, height);

    auto expected = filter(mat, width, coreSize);
    auto calculated = filterParallel(mat, width, coreSize);

    ASSERT_EQ(expected, calculated);
}

TEST(STD_Gaussian_blur, PerformScheduling_Throws_On_Invalid_Width) {
    const int height = 30;
    const int coreRadius = 2;
    const int numThreads = 5;

    ASSERT_ANY_THROW(performScheduling(-7, height, coreRadius, numThreads));
    ASSERT_ANY_THROW(performScheduling(0, height, coreRadius, numThreads));
}

TEST(STD_Gaussian_blur, PerformScheduling_Throws_On_Invalid_Height) {
    const int width = 30;
    const int coreRadius = 2;
    const int numThreads = 5;

    ASSERT_ANY_THROW(performScheduling(width, -4, coreRadius, numThreads));
    ASSERT_ANY_THROW(performScheduling(width, 0, coreRadius, numThreads));
}

TEST(STD_Gaussian_blur, PerformScheduling_Throws_On_Invalid_CoreRadius) {
    const int width = 27;
    const int height = 30;
    const int numThreads = 5;

    ASSERT_ANY_THROW(performScheduling(width, height, -11, numThreads));
    ASSERT_ANY_THROW(performScheduling(height, height, 0, numThreads));
}

TEST(STD_Gaussian_blur, PerformScheduling_Throws_On_Invalid_Number_Of_Threads) {
    const int width = 27;
    const int height = 30;
    const int coreRadius = 2;

    ASSERT_ANY_THROW(performScheduling(width, height, coreRadius, 0));
    ASSERT_ANY_THROW(performScheduling(width, height, coreRadius, -8));
}

TEST(STD_Gaussian_blur, PerformScheduling_No_Throw_On_Valid_Args) {
    const int width = 27;
    const int height = 30;
    const int coreRadius = 2;
    const int numThreads = 5;

    ASSERT_NO_THROW(performScheduling(width, height, coreRadius, numThreads));
}

TEST(STD_Gaussian_blur, PerformScheduling_Returns_Vector_Of_Valid_Size) {
    const int width = 42;
    const int height = 17;
    const int coreRadius = 4;
    const int numThreads = 11;

    auto schedule = performScheduling(width, height, coreRadius, numThreads);

    ASSERT_EQ(schedule.size(), schedule.size());
}

// TEST(STD_Gaussian_blur, FilterParallel_Accelerates) {
//    int width = 20000, height = 20000;
//    auto mat = getRandomMatrix(width, height);
//
//    auto start = std::chrono::high_resolution_clock::now();
//    filter(mat, width, 9);
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> sequential = end - start;
//
//    start = std::chrono::high_resolution_clock::now();
//    filterParallel(mat, width, 9);
//    end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> parallel = end - start;
//
//    ASSERT_GT(sequential, parallel);
//
//    std::cout << "Sequential: " << sequential.count() << std::endl;
//    std::cout << "Parallel: " << parallel.count() << std::endl;
//    std::cout << "Scale: " << sequential / parallel << std::endl;
// }
