// Copyright 2021 Tarasov Nikita
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include "../../../modules/task_2/tarasov_n_gaussian_filter/gaussian_filter.h"

TEST(Gaussian_filter, Test1) {
    ASSERT_ANY_THROW(get_random_img(0, 1));
}

TEST(Gaussian_filter, Test2) {
    int height = 10;
    int width = 10;
    std::vector<double> img(height * 3);
    ASSERT_ANY_THROW(gaussian_filter(img, width, height, 1.0));
}

TEST(Gaussian_filter, Test3) {
    int height = 2;
    int width = 2;
    std::vector<double> img(width * height);
    ASSERT_ANY_THROW(gaussian_filter(img, width, height, 1.0));
}

TEST(Gaussian_filter, Test4) {
    std::vector<double> empty_img(13);
    ASSERT_ANY_THROW(gaussian_filter(empty_img, 13, 13, 1.0));
}

TEST(Gaussian_filter, Test5) {
    std::vector<double> img = get_random_img(401, 201);
    ASSERT_NO_THROW(gaussian_filter(img, 401, 201, 1.0));
}

TEST(Gaussian_filter, Test6) {
    std::vector<double> core;
    core = gaussian_core(1.0);
    double sum = 0;
    for (int i = 0; i < 9; i++) {
        sum += core[i];
    }
    ASSERT_EQ(sum, 1);
}

TEST(Gaussian_filter, Test7) {
    std::vector<double> img = {
        71.0, 73.0, 177.0, 24.0, 46.0,
        168.0, 192.0, 125.0, 79.0, 19.0,
        127.0, 228.0, 24.0, 55.0, 215.0,
        200.0, 170.0, 230.0, 113.0, 48.0,
        140.0, 163.0, 180.0, 77.0, 124.0
    };

    std::vector<double> expectation = {
        99.785, 119.072, 113.194, 72.839, 38.73,
        140.887, 142.735, 112.519, 78.449, 73.877,
        170.301, 164.391, 125.524, 96.303, 105.088,
        169.148, 171.763, 146.557, 114.634, 107.1,
        158.767, 170.485, 156.534, 120.616, 98.702
    };

    img = gaussian_filter_omp(img, 5, 5, 1.0);

    for (int i = 0; i < 25; i++) {
        ASSERT_DOUBLE_EQ(img[i], expectation[i]);
    }
}

TEST(Gaussian_filter, Test8) {
    double sigma = 1.0;
    double t1, t2;

    int width = 10000;
    int height = 10000;

    std::vector<double> random_matrix = get_random_img(width, height);
    t1 = omp_get_wtime();
    std::vector<double> seq_result = gaussian_filter(random_matrix, width, height, sigma);
    t2 = omp_get_wtime();
    std::cout << "Sequential time: " << t2 - t1 << std::endl;

    t1 = omp_get_wtime();
    std::vector<double> omp_result = gaussian_filter_omp(random_matrix, width, height, sigma);
    t2 = omp_get_wtime();
    std::cout << "OpenMP time: " << t2 - t1 << std::endl;

    ASSERT_EQ(seq_result, omp_result);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
