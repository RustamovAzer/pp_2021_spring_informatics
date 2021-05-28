// Copyright 2021 Tarasov Nikita
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/tarasov_n_gaussian_filter/gaussian_filter.h"

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
        71.0, 73.0, 177.0, 24.0, 46.0,
        168.0, 142.735, 112.519, 78.449, 19.0,
        127.0, 164.391, 125.524, 96.303, 215.0,
        200.0, 171.763, 146.557, 114.634, 48.0,
        140.0, 163.0, 180.0, 77.0, 124.0
    };
    img = gaussian_filter(img, 5, 5, 1.0);
    for (int i = 0; i < 25; i++) {
        ASSERT_DOUBLE_EQ(img[i], expectation[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
