// Copyright 2021 Kirichenko Nikita
#include <gtest/gtest.h>
#include "./contrast.h"

TEST(Image_contrast, Test1) {
    int l = 2;
    int k = 2;
    Result image = { 0, 3, 3, 0 };
    Result count = { 0, 255, 255, 0 };
    Result result = Contrasttbb(image);
    for (int i = 0; i < l * k; i++) {
        ASSERT_EQ(result[i], count[i]);
    }
}
TEST(Image_contrast, Test2) {
    int l = 3;
    int k = 3;
    Result image = { 0, 3, 5, 7, 9, 6, 8, 10, 14 };
    Result count = { 0, 54, 91, 127, 163, 109, 145, 182, 255 };
    Result result = Contrasttbb(image);
    for (int i = 0; i < l * k; i++) {
        ASSERT_EQ(result[i], count[i]);
    }
}
TEST(Image_contrast, Test3) {
    int l = 4;
    int k = 4;
    Result image = { 0, 1, 5, 14, 16, 12, 6, 9, 19, 25,
        35, 43, 67, 78, 45, 96 };
    Result count = { 0, 2, 13, 37, 42, 31, 15, 23, 50,
        66, 92, 114, 177, 207, 119, 255 };
    Result result = Contrasttbb(image);
    for (int i = 0; i < l * k; i++) {
        ASSERT_EQ(result[i], count[i]);
    }
}
TEST(Image_contrast, Test4) {
    int l = 5;
    int k = 5;
    Result image = { 0, 26, 43, 25, 68, 29, 78, 95, 45, 23, 46, 28,
        69, 20, 30, 46, 47, 67, 79, 81, 40, 18, 15, 75, 39, 99 };
    Result count = { 0, 66, 110, 64, 175, 74, 200, 244, 115, 59, 118, 72,
        177, 51, 77, 118, 121, 172, 203, 208, 103, 46, 38, 193, 100, 255 };
    Result result = Contrasttbb(image);
    for (int i = 0; i < l * k; i++) {
        ASSERT_EQ(result[i], count[i]);
    }
}
TEST(Image_contrast, DISABLED_Test5) {
    int l = 10000;
    int k = 10000;
    Result image = RandomI(l, k);
    tbb::tick_count time1 = tbb::tick_count::now();
    Result result_seq = Contrast(image);
    tbb::tick_count time2 = tbb::tick_count::now();
    std::cout << "Seq: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
    time1 = tbb::tick_count::now();
    Result result_tbb = Contrasttbb(image);
    time2 = tbb::tick_count::now();
    std::cout << "Tbb: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
        ASSERT_EQ(result_seq, result_tbb);
    }
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
