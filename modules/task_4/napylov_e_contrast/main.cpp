// Copyright 2021 Napylov Evgenii
#include <gtest/gtest.h>
#include <time.h>
#include <iostream>
#include "./contrast.h"

TEST(Linear_stretch_contrast_STD, Manual_Calc_1) {
     VecImage image = {0, 1, 1, 1, 0, 1 };
     VecImage correct = { 0, 255, 255, 255, 0, 255 };
     VecImage res = add_contrast_std(image);
     for (size_t i = 0; i < image.size(); i++) {
         ASSERT_EQ(correct[i], res[i]);
     }
}

TEST(Linear_stretch_contrast_STD, Manual_Calc_2) {
     VecImage image = { 5, 123, 28, 240 };
     VecImage correct = { 0, 128, 25, 255 };
     VecImage res = add_contrast_std(image);
     for (size_t i = 0; i < image.size(); i++) {
         ASSERT_EQ(correct[i], res[i]);
     }
}

TEST(Linear_stretch_contrast_STD, Manual_Calc_3) {
     VecImage image = { 57, 120, 99, 32, 17, 64, 1, 55, 200 };
     VecImage correct = { 72, 152, 126, 40, 21, 81, 0, 69, 255 };
     VecImage res = add_contrast_std(image);
     for (size_t i = 0; i < image.size(); i++) {
         ASSERT_EQ(correct[i], res[i]);
     }
}

TEST(Linear_stretch_contrast_STD, Manual_Calc_4) {
     VecImage image = { 1, 2, 3, 4, 5, 1 };
     VecImage correct = { 0, 64, 128, 191, 255, 0 };
     VecImage res = add_contrast_std(image);
     for (size_t i = 0; i < image.size(); i++) {
         ASSERT_EQ(correct[i], res[i]);
     }
}

TEST(Linear_stretch_contrast_STD, Seq_vs_Std) {
    VecImage image = RandomVector(1920 * 1080);

    clock_t t1 = clock();
    VecImage res_seq = add_contrast(image);
    clock_t t2 = clock();
    double t_seq = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC;

    t1 = clock();
    VecImage res_std = add_contrast_std(image);
    t2 = clock();
    double t_std = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC;

    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(res_seq[i], res_std[i]);
    }

    std::cout << "seq_t: " << t_seq << "\nstd_t: " << t_std << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
