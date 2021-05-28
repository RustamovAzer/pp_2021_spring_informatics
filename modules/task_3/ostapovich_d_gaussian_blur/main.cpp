// Copyright 2021 Denis Ostapovich
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include "./gaussian_blur.h"

TEST(TBB_Gaussian_blur, GetRandomMatrix_Throws_On_Zero_Args) {
    ASSERT_ANY_THROW(getRandomMatrix(0, 0));
}

TEST(TBB_Gaussian_blur, GetRandomMatrix_Throws_On_Negative_Args) {
    ASSERT_ANY_THROW(getRandomMatrix(-1, -1));
}

TEST(TBB_Gaussian_blur, GetRandomMatrix_No_Throw_On_Valid_Args) {
    ASSERT_NO_THROW(getRandomMatrix(3, 5));
}

TEST(TBB_Gaussian_blur, GetRandomMatrix_Returns_Vector_Of_Valid_Size) {
    auto vec = getRandomMatrix(2, 7);
    ASSERT_EQ(vec.size(), (size_t)14);
}

TEST(TBB_Gaussian_blur, CalculateCore_Returns_Valid_Core) {
    int size = 5;
    auto core = calculateCore(size);
    for (double num : core) {
        ASSERT_NE(num, 0);
    }
}

TEST(TBB_Gaussian_blur, Filter_Throws_On_Zero_Size_Matrix) {
    std::vector<uint8_t> mat;
    ASSERT_ANY_THROW(filter(mat, 10));
}

TEST(TBB_Gaussian_blur, Filter_Throws_On_Zero_Or_Negative_Width_Value) {
    auto mat = getRandomMatrix(11, 21);
    ASSERT_ANY_THROW(filter(mat, 0));
    ASSERT_ANY_THROW(filter(mat, -1));
}

TEST(TBB_Gaussian_blur, Filter_Throws_Invalid_Width_Value) {
    auto mat = getRandomMatrix(11, 21);
    ASSERT_ANY_THROW(filter(mat, 10));
}

TEST(TBB_Gaussian_blur, Filter_Throws_On_Invalid_Core_Size) {
    int width = 11;
    auto mat = getRandomMatrix(11, 21);
    ASSERT_ANY_THROW(filter(mat, width, 0));
    ASSERT_ANY_THROW(filter(mat, width, -1));
    ASSERT_ANY_THROW(filter(mat, width, 4));
}

TEST(TBB_Gaussian_blur, Filter_No_Throw_On_Valid_Args) {
    int width = 21;
    int coreSize = 3;
    auto mat = getRandomMatrix(width, 31);
    ASSERT_NO_THROW(filter(mat, width, coreSize));
}

TEST(TBB_Gaussian_blur, FilterParallel_Throws_On_Zero_Size_Matrix) {
    std::vector<uint8_t> mat;
    ASSERT_ANY_THROW(filterParallel(mat, 10));
}

TEST(TBB_Gaussian_blur, FilterParallel_Throws_On_Zero_Or_Negative_Width_Value) {
    auto mat = getRandomMatrix(11, 21);
    ASSERT_ANY_THROW(filterParallel(mat, 0));
    ASSERT_ANY_THROW(filterParallel(mat, -1));
}

TEST(TBB_Gaussian_blur, FilterParallel_Throws_Invalid_Width_Value) {
    auto mat = getRandomMatrix(10, 20);
    ASSERT_ANY_THROW(filterParallel(mat, 3));
}

TEST(TBB_Gaussian_blur, FilterParallel_Throws_On_Invalid_Core_Size) {
    auto mat = getRandomMatrix(11, 21);
    ASSERT_ANY_THROW(filterParallel(mat, 11, 0));
    ASSERT_ANY_THROW(filterParallel(mat, 11, -5));
    ASSERT_ANY_THROW(filterParallel(mat, 11, 8));
}

TEST(TBB_Gaussian_blur, FilterParallel_No_Throw_On_Valid_Args) {
    auto mat = getRandomMatrix(10, 20);
    ASSERT_NO_THROW(filterParallel(mat, 10));
}

TEST(TBB_Gaussian_blur, FilterParallel_Result_Equals_To_Filter_Result) {
    int width = 31;
    int coreSize = 5;
    auto mat = getRandomMatrix(width, 44);
    auto expected = filter(mat, width, coreSize);
    auto calculated = filterParallel(mat, width, coreSize);
    ASSERT_EQ(expected, calculated);
}

// TEST(TBB_Gaussian_blur, FilterParallel_Accelerates) {
//    double sequential, parallel;
//    int width = 1000, height = 1000;
//    auto mat = getRandomMatrix(width, height);

//    double start, end;
//    start = omp_get_wtime();
//    filter(mat, width, 9);
//    end = omp_get_wtime();
//    sequential = end - start;

//    start = omp_get_wtime();
//    filterParallel(mat, width, 9);
//    end = omp_get_wtime();
//    parallel = end - start;

//    ASSERT_GT(sequential, parallel);
//    std::cout << "Sequential: " << sequential << std::endl;
//    std::cout << "Parallel: " << parallel << std::endl;
//    std::cout << "Scale: " << sequential / parallel << std::endl;
// }
