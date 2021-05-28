// Copyright 2020 Lyutova Tanya
#include <gtest/gtest.h>
#include <omp.h>
#include "../../modules/task_2/lyutova_t_sobel/lyutova_t_sobel.h"

TEST(OMP_Sobel, RandomImage) {
    Image image = createRandomImage(100, 100);
    ASSERT_EQ(image.cols, 100);
    ASSERT_EQ(image.rows, 100);
}

TEST(OMP_Sobel, SameAsSequential) {
  Image image = createRandomImage(100, 100);
  Image seq = sobel(image);
  Image par = sobelParallel(image);
  ASSERT_EQ(seq, par);
}

TEST(OMP_Sobel, DifferentImages) {
    Image image = createRandomImage(100, 100);
    Image res = sobelParallel(image);
    ASSERT_NE(image.pixels, res.pixels);
}

TEST(OMP_Sobel, SameResult) {
    Image image(4, 4);
    Image seq = sobel(image);
    Image par = sobelParallel(image);
    ASSERT_EQ(seq, par);
}

TEST(OMP_Sobel, Different) {
    Image image = createRandomImage(100, 100);
    Image image2(100, 100);
    ASSERT_NE(image.pixels, image2.pixels);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
