// Copyright 2021 Napylov Evgenii
#include <gtest/gtest.h>
#include "./contrast.h"

TEST(Linear_stretch_contrast, Manual_Calc_1) {
    int w = 3; int h = 2;
    Image image = { {0, 1}, {1, 1}, {0, 1} };
    Image correct = { {0, 255}, {255, 255}, {0, 255} };
    Image res = vec_to_image(add_contrast(image_to_vec(image, w, h)), w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            ASSERT_EQ(correct[i][j], res[i][j]);
        }
    }
}

TEST(Linear_stretch_contrast, Manual_Calc_2) {
    int w = 2; int h = 2;
    Image image = { {5, 123}, {28, 240} };
    Image correct = { {0, 128}, {25, 255} };
    Image res = vec_to_image(add_contrast(image_to_vec(image, w, h)), w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            ASSERT_EQ(correct[i][j], res[i][j]);
        }
    }
}

TEST(Linear_stretch_contrast, Manual_Calc_3) {
    int w = 3; int h = 3;
    Image image = { {57, 120, 99}, {32, 17, 64}, {1, 55, 200} };
    Image correct = { {72, 152, 126}, {40, 21, 81}, {0, 69, 255} };
    Image res = vec_to_image(add_contrast(image_to_vec(image, w, h)), w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            ASSERT_EQ(correct[i][j], res[i][j]);
        }
    }
}

TEST(Linear_stretch_contrast, Manual_Calc_4) {
    int w = 3; int h = 2;
    Image image = { {1, 2}, {3, 4}, {5, 1} };
    Image correct = { {0, 64}, {128, 191}, {255, 0} };
    Image res = vec_to_image(add_contrast(image_to_vec(image, w, h)), w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            ASSERT_EQ(correct[i][j], res[i][j]);
        }
    }
}

TEST(Linear_stretch_contrast, Random_No_Throw) {
    VecImage image = RandomVector(1920 * 1080);
    VecImage res;
    ASSERT_NO_THROW(res = add_contrast(image));
}
