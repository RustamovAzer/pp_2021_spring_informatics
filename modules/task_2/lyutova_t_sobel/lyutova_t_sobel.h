// Copyright 2020 Lyutova Tanya
#ifndef MODULES_TASK_2_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_
#define MODULES_TASK_2_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_

#include <cmath>
#include <vector>

struct Image {
  std::vector<int> pixels;
  int rows = 0;
  int cols = 0;

  Image(const std::vector<int> &pixels, int rows, int cols)
      : pixels(pixels), rows(rows), cols(cols) {}

  Image(int rows, int cols) : pixels(rows * cols, 0), rows(rows), cols(cols) {}
  const int &operator()(int x, int y) const { return pixels[x * cols + y]; }
  int &operator()(int x, int y) { return pixels[x * cols + y]; }

  bool operator==(const Image &_img) const {
    if (rows == _img.rows && cols == _img.cols)
      return true;
    return false;
  }
};

Image createRandomImage(int rows, int cols);
Image sobel(const Image &image);
Image sobelParallel(const Image &image);

#endif  // MODULES_TASK_2_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_
