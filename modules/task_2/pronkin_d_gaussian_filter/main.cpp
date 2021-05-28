// Copyright 2021 Pronkin Dmitry
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include "./gaussian_filter.h"

TEST(Par_Gaussian_Filter, Image_256_256) {
    std::vector<double> image = getRandomImage(256, 256);
    std::vector<double> test = seqGaussianFilter(image, 256, 256, 1.0);
    std::vector<double> res = parGaussianFilter(image, 256, 256, 1.0);
    ASSERT_EQ(test, res);
}

TEST(Par_Gaussian_Filter, Image_1_256) {
    std::vector<double> image = getRandomImage(1, 256);
    std::vector<double> test = seqGaussianFilter(image, 1, 256, 1.0);
    std::vector<double> res = parGaussianFilter(image, 1, 256, 1.0);
    ASSERT_EQ(test, res);
}

TEST(Par_Gaussian_Filter, Image_256_1) {
    std::vector<double> image = getRandomImage(256, 1);
    std::vector<double> test = seqGaussianFilter(image, 256, 1, 1.0);
    std::vector<double> res = parGaussianFilter(image, 256, 1, 1.0);
    ASSERT_EQ(test, res);
}

TEST(Par_Gaussian_Filter, Image_513_123) {
    std::vector<double> image = getRandomImage(513, 123);
    std::vector<double> test = seqGaussianFilter(image, 513, 123, 1.0);
    std::vector<double> res = parGaussianFilter(image, 513, 123, 1.0);
    ASSERT_EQ(test, res);
}

TEST(Par_Gaussian_Filter, Image_1920_1080) {
    std::vector<double> image = getRandomImage(1920, 1080);
    std::vector<double> test = seqGaussianFilter(image, 1920, 1080, 1.0);
    std::vector<double> res = parGaussianFilter(image, 1920, 1080, 1.0);
    ASSERT_EQ(test, res);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
