// Copyright 2021 Pronkin Dmitry
#include <gtest/gtest.h>
#include <vector>
#include "./gaussian_filter.h"

TEST(Seq_Gaussian_Filter, Throw_Empty_Image) {
    std::vector<double> emptyImage;
    ASSERT_ANY_THROW(seqGaussianFilter(emptyImage, 13, 13, 1.0));
}

TEST(Seq_Gaussian_Filter, Throw_Incorrect_Image_Size) {
    std::vector<double> emptyImage(13);
    ASSERT_ANY_THROW(seqGaussianFilter(emptyImage, -13, -13, 1.0));
}

TEST(Seq_Gaussian_Filter, Throw_Size_Missmatch) {
    std::vector<double> emptyImage(13);
    ASSERT_ANY_THROW(seqGaussianFilter(emptyImage, 13, 13, 1.0));
}

TEST(Seq_Gaussian_Filter, Working_Capacity) {
    std::vector<double> image = getRandomImage(640, 480);
    ASSERT_NO_THROW(seqGaussianFilter(image, 640, 480, 1.0));
}

TEST(Seq_Gaussian_Filter, Image_5_5) {
    std::vector<double> image = {
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.2, 0.4, 0.2, 0.1,
        0.1, 0.6, 0.8, 0.6, 0.1,
        0.1, 0.2, 0.4, 0.2, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1
    };
    std::vector<double> control = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.244664, 0.320418, 0.244664, 0.0,
        0.0, 0.358615, 0.471118, 0.358615, 0.0,
        0.0, 0.2446640, 0.320418, 0.244664, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0
    };
    std::vector<double> result = seqGaussianFilter(image, 5, 5, 1.0);
    for (int i = 0; i < 25; i++) {
        ASSERT_NEAR(control[i], result[i], 0.000001);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
