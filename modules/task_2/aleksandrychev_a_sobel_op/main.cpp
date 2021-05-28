// Copyright 2021 Aleksandrychev Andrey
#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "./sobel_op.h"

TEST(ConvexHull, createImageTest1) {
    Matrix image, sobel_image;
    int rows = 5;
    int cols = 5;
    image = create_binary_image(rows, cols);
    sobel_image = sobel_op(image, rows, cols);

    ASSERT_DOUBLE_EQ(12, 12);
}

TEST(ConvexHull, createImageTest2) {
    Matrix image, sobel_image;
    int rows = 5;
    int cols = 5;
    image = create_binary_image(rows, cols);
    sobel_image = sobel_op(image, rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << image[i][j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "\n";

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << sobel_image[i][j] << "\t";
        }
        std::cout << "\n";
    }

    ASSERT_DOUBLE_EQ(12, 12);
}

TEST(ConvexHull, createImageTest3) {
    Matrix image, sobel_image;
    int rows = 5;
    int cols = 5;
    image = create_binary_image(rows, cols);
    sobel_image = sobel_op(image, rows, cols);

    ASSERT_DOUBLE_EQ(12, 12);
}

TEST(ConvexHull, createImageTest4) {
    Matrix image, sobel_image;
    int rows = 7;
    int cols = 5;
    image = create_binary_image(rows, cols);
    sobel_image = sobel_op(image, rows, cols);

    ASSERT_DOUBLE_EQ(12, 12);
}

TEST(ConvexHull, createImageTest5) {
    Matrix image, sobel_image;
    int rows = 5;
    int cols = 7;
    image = create_binary_image(rows, cols);
    sobel_image = sobel_op(image, rows, cols);

    ASSERT_DOUBLE_EQ(12, 12);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
