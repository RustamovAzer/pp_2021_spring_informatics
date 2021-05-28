// Copyright 2020 Lyutova Tanya
#ifndef MODULES_TASK_1_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_
#define MODULES_TASK_1_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_

#include <cmath>
#include <vector>

struct Image {
    std::vector<int> pixels;
    int rows = 0;
    int cols = 0;

    Image(const std::vector<int>& pixels, int rows, int cols) : pixels(pixels), rows(rows), cols(cols) {
    }

    Image(int rows, int cols) : pixels(rows * cols, 0), rows(rows), cols(cols) {
    }
    const int& operator()(int x, int y) const {
        return pixels[x * cols + y];
    }
    int& operator()(int x, int y) {
        return pixels[x * cols + y];
    }

    bool operator == (const Image& _img) const {
        if (rows == _img.rows && cols == _img.cols)
            return true;
        return false;
    }
};

inline int clamp(int value, int min, int max) {
    if (value > max)
        return max;
    if (value < min)
        return min;
    return value;
}

#endif  // MODULES_TASK_1_LYUTOVA_T_SOBEL_LYUTOVA_T_SOBEL_H_
