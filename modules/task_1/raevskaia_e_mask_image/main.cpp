// Copyright 2021 Ekaterina Raevskaya

#include <gtest/gtest.h>
#include "./mask_image.h"

TEST(mask_image, Test_Mask_image_1) {
    int n = 4, m = 4;
    std::vector <int> a = {
        1, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 0, 1,
        0, 1, 0, 1
    };
    std::vector <int> ans = {
        1, 0, 0, 0,
        0, 2, 2, 0,
        0, 2, 0, 3,
        0, 2, 0, 3
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_2) {
    int n = 4, m = 4;
    std::vector <int> a = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    std::vector <int> ans = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_3) {
    int n = 5, m = 4;
    std::vector <int> a = {
        1, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 0, 1,
        0, 1, 0, 1,
        1, 0, 1, 0
    };
    std::vector <int> ans = {
        1, 0, 0, 0,
        0, 2, 2, 0,
        0, 2, 0, 3,
        0, 2, 0, 3,
        4, 0, 5, 0
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_4) {
    int n = 5, m = 6;
    std::vector <int> a = {
        1, 0, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 1
    };
    std::vector <int> ans = {
        1, 0, 0, 0, 2, 0,
        0, 3, 3, 0, 2, 0,
        0, 3, 0, 2, 2, 2,
        0, 3, 0, 2, 0, 0,
        4, 0, 5, 0, 6, 6
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_5) {
    int n = 5, m = 6;
    std::vector <int> a = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        1, 1, 0, 1, 0, 0,
        1, 0, 1, 1, 1, 1
    };
    std::vector <int> ans = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        1, 1, 0, 1, 0, 0,
        1, 0, 1, 1, 1, 1
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_6) {
    int n = 5, m = 6;
    std::vector <int> a = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1
    };
    std::vector <int> ans = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        2, 0, 3, 3, 3, 3
    };
    ASSERT_EQ(mask(&a, n, m), ans);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
