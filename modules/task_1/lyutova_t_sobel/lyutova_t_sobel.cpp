// Copyright 2020 Lyutova Tanya
#include "../../modules/task_1/lyutova_t_sobel/lyutova_t_sobel.h"

Image sobel(const Image& image) {
    static const std::vector<int> matrixX = { -1, 2, -1, 0, 0, 0, 1, 2, 1 };
    static const std::vector<int> matrixY = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int matrixSize = 3;
    static const int matrixRadius = 1;
    Image filteredImage(image.rows, image.cols);
    for (int x = 0; x < image.rows; x++)
        for (int y = 0; y < image.cols; y++) {
            int valX = 0, valY = 0;
            for (int i = -matrixRadius; i <= matrixRadius; i++)
                for (int j = -matrixRadius; j <= matrixRadius; j++) {
                    int index = (i + matrixRadius) * matrixSize + j + matrixRadius;
                    int nearX = clamp(x + j, 0, image.rows - 1);
                    int nearY = clamp(y + i, 0, image.cols - 1);
                    valX += image(nearX, nearY) * matrixX[index];
                    valY += image(nearX, nearY) * matrixY[index];
                }
            filteredImage(x, y) = clamp(static_cast<int>(std::sqrt(valX * valX + valY * valY)), 0, 255);
        }
    return filteredImage;
}
