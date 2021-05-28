// Copyright 2021 Pronkin Dmitry
#include <gtest/gtest.h>
#include <vector>
#include "./gaussian_filter.h"

TEST(GaussianFilter, Image_256_256) {
    int height = 256, width = 256;
    std::vector<double> image = getRandomImage(height, width);
    std::vector<double> test = seqGaussianFilter(image, height, width, 1.0);
    std::vector<double> img = parGaussianFilter(image, height, width, 1.0);
    ASSERT_EQ(test, img);
}

TEST(GaussianFilter, Image_113_113) {
    int height = 113, width = 113;
    std::vector<double> image = getRandomImage(height, width);
    std::vector<double> test = seqGaussianFilter(image, height, width, 1.0);
    std::vector<double> img = parGaussianFilter(image, height, width, 1.0);
    ASSERT_EQ(test, img);
}

TEST(GaussianFilter, Image_257_1) {
    int height = 257, width = 1;
    std::vector<double> image = getRandomImage(height, width);
    std::vector<double> test = seqGaussianFilter(image, height, width, 1.0);
    std::vector<double> img = parGaussianFilter(image, height, width, 1.0);
    ASSERT_EQ(test, img);
}

TEST(GaussianFilter, Image_1_257) {
    int height = 1, width = 257;
    std::vector<double> image = getRandomImage(height, width);
    std::vector<double> test = seqGaussianFilter(image, height, width, 1.0);
    std::vector<double> img = parGaussianFilter(image, height, width, 1.0);
    ASSERT_EQ(test, img);
}

TEST(GaussianFilter, Image_1080_1920) {
    int height = 1080, width = 1920;
    std::vector<double> image = getRandomImage(height, width);
    std::vector<double> test = seqGaussianFilter(image, height, width, 1.0);
    std::vector<double> img = parGaussianFilter(image, height, width, 1.0);
    ASSERT_EQ(test, img);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
