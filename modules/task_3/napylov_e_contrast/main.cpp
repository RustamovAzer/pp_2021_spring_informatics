// Copyright 2021 Napylov Evgenii
#include <tbb/tick_count.h>
#include <gtest/gtest.h>
#include <iostream>

#include "./contrast.h"


TEST(Linear_stretch_contrast_TBB, Manual_Calc_1) {
    VecImage image = {0, 1, 1, 1, 0, 1 };
    image = add_contrast_tbb(image);
    VecImage correct = { 0, 255, 255, 255, 0, 255 };
    VecImage res = add_contrast_tbb(image);
    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(correct[i], res[i]);
    }
}

TEST(Linear_stretch_contrast_TBB, Manual_Calc_2) {
    VecImage image = { 5, 123, 28, 240 };
    VecImage correct = { 0, 128, 25, 255 };
    VecImage res = add_contrast_tbb(image);
    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(correct[i], res[i]);
    }
}

TEST(Linear_stretch_contrast_TBB, Manual_Calc_3) {
    VecImage image = { 57, 120, 99, 32, 17, 64, 1, 55, 200 };
    VecImage correct = { 72, 152, 126, 40, 21, 81, 0, 69, 255 };
    VecImage res = add_contrast_tbb(image);
    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(correct[i], res[i]);
    }
}

TEST(Linear_stretch_contrast_TBB, Manual_Calc_4) {
    VecImage image = { 1, 2, 3, 4, 5, 1 };
    VecImage correct = { 0, 64, 128, 191, 255, 0 };
    VecImage res = add_contrast_tbb(image);
    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(correct[i], res[i]);
    }
}

TEST(Linear_stretch_contrast_TBB, Seq_vs_Tbb) {
    VecImage image = RandomVector(1920 * 1080);
    tbb::tick_count t0, t1;
    double dt_seq, dt_tbb;

    t0 = tbb::tick_count::now();
    VecImage res_seq = add_contrast(image);
    t1 = tbb::tick_count::now();
    dt_seq = (t1 - t0).seconds();

    t0 = tbb::tick_count::now();
    VecImage res_tbb = add_contrast_tbb(image);
    t1 = tbb::tick_count::now();
    dt_tbb = (t1 - t0).seconds();

    for (size_t i = 0; i < image.size(); i++) {
        ASSERT_EQ(res_seq[i], res_tbb[i]);
    }

    std::cout << "seq_t: " << dt_seq << std::endl;
    std::cout << "tbb_t: " << dt_tbb << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
