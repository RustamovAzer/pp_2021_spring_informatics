// Copyright 2020 Lyutova Tanya
#include "../../modules/task_1/lyutova_t_sobel/lyutova_t_sobel.h"
#include "gtest/gtest.h"

TEST(sequential, EmptyMatrix) {
    Image tmp(0, 0);
    ASSERT_EQ(tmp.cols, 0);
    ASSERT_EQ(tmp.rows, 0);
}

TEST(sequential, Create) {
    Image tmp(1, 1);
    ASSERT_EQ(tmp.cols, 1);
    ASSERT_EQ(tmp.rows, 1);
}

TEST(sequential, CopyImage) {
    Image tmp(1, 1);
    Image tmp2{ tmp };
    ASSERT_EQ(tmp, tmp2);
}

TEST(sequential, Equal) {
    Image tmp(1, 1);
    ASSERT_EQ(tmp, tmp);
}

TEST(sequential, CompareSameImage) {
    Image tmp(1, 1);
    Image tmp2(1, 1);
    ASSERT_EQ(tmp, tmp2);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
