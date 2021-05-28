// Copyright 2021 Denis Ostapovich
#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <random>
#include "../../../modules/task_4/ostapovich_d_gaussian_blur/gaussian_blur.h"
#include "../../../3rdparty/unapproved/unapproved.h"

vec filter(const vec& matrix, int width, int coreSize) {
    std::size_t length = matrix.size();

    if (length == 0 || width <= 0 || length % width != 0 || coreSize < 3 || coreSize % 2 == 0) {
        throw std::invalid_argument("Zero or negative argument passed");
    }

    int radius = coreSize / 2;
    auto core = calculateCore(coreSize);

    int height = length / width;
    std::vector<uint8_t> filtered(matrix);

    for (int i = radius; i < height - radius; i++) {
        for (int j = radius; j < width - radius; j++) {
            uint8_t result = 0;
            for (int m = -radius; m <= radius; m++) {
                for (int n = -radius; n <= radius; n++) {
                    result += matrix[width * (i + m) + j + n]
                        * core[coreSize * (m + radius) + n + radius];
                }
            }
            filtered[width * i + j] = result;
        }
    }

    return filtered;
}

vec filterParallel(const vec& matrix, int width, int coreSize) {
    const int numThreads = std::thread::hardware_concurrency();
    const std::size_t length = matrix.size();

    if (length == 0 || width <= 0 || length % width != 0 || coreSize < 3 || coreSize % 2 == 0) {
        throw std::invalid_argument("Zero or negative argument passed");
    }

    const int coreRadius = coreSize / 2;
    const int height = length / width;

    const std::vector<double> core = calculateCore(coreSize);
    vec filtered(matrix);

    auto schedule = performScheduling(width, height, coreRadius, numThreads);

    if (schedule.empty()) {
        return filter(matrix, width, coreSize);
    }

    std::thread* threads = new std::thread[numThreads];

    for (int i = 0; i < numThreads; i++) {
        threads[i] = std::thread([&](int index) {
            auto chunks = schedule[index];
            for (size_t l = 0; l < chunks.size(); l++) {
                auto chunk = chunks[l];
                int startRow = chunk.topLeftCorner.first;
                int startCol = chunk.topLeftCorner.second;
                int endRow = chunk.bottomRightCorner.first;
                int endColumn = chunk.bottomRightCorner.second;

                for (int j = startRow; j <= endRow; j++) {
                    for (int k = startCol; k <= endColumn; k++) {
                        uint8_t result = 0;
                        for (int m = -coreRadius; m <= coreRadius; m++) {
                            for (int n = -coreRadius; n <= coreRadius; n++) {
                                result += matrix[width * (j + m) + k + n]
                                    * core[coreSize * (m + coreRadius) + n + coreRadius];
                            }
                        }
                        filtered[width * j + k] = result;
                    }
                }
            }
            }, i);
    }

    for (int i = 0; i < numThreads; i++) {
        threads[i].join();
    }

    delete[]threads;

    return filtered;
}

std::vector<std::vector<Chunk>> performScheduling(int width, int height, int coreRadius, int numThreads) {
    if (width <= 0 || height <= 0 || coreRadius < 1 || numThreads <= 0) {
        throw std::invalid_argument("Zero or negative argument passed");
    }

    std::vector<std::vector<Chunk>> schedule;

    const int rowIterations = height - 2 * coreRadius;
    const int columnIterations = width - 2 * coreRadius;

    const int rowDelta = rowIterations / numThreads;
    const int rowRemainder = rowIterations % numThreads;
    const int columnDelta = columnIterations / numThreads;
    const int columnRemainder = columnIterations % numThreads;

    if (rowDelta == 0 && columnDelta == 0) {
        return schedule;
    }

    const int chunksInHeight = rowDelta != 0 ? numThreads : 1;
    const int chunksInWidth = columnDelta != 0 ? numThreads : 1;

    schedule.assign(numThreads, std::vector<Chunk>());

    for (int i = 0; i < chunksInHeight; i++) {
        for (int j = 0; j < chunksInWidth; j++) {
            int startRow = coreRadius + rowDelta * i;
            int startCol = coreRadius + columnDelta * j;
            int endRow = startRow + rowDelta - 1;
            int endCol = startCol + columnDelta - 1;
            if (i == chunksInHeight - 1)
                endRow += rowRemainder;
            if (j == chunksInWidth - 1)
                endCol += columnRemainder;

            int currentThread = (i + j) % numThreads;
            schedule.at(currentThread).push_back(
                Chunk{ std::pair<int, int>(startRow, startCol),
                std::pair<int, int>(endRow, endCol) });
        }
    }

    return schedule;
}

std::vector<double> calculateCore(int size, double deviation) {
    if (size < 3) {
        throw std::out_of_range("Size must be greater or equal to 3");
    }
    if (size % 2 == 0) {
        throw std::invalid_argument("Size must be an odd number");
    }
    double sum = 0;
    double s = 2.0 * deviation * deviation;
    int radius = size / 2;
    std::vector<double> core(size * size);

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            double res = (std::exp(-(i * i + j * j) / s)) / (M_PI * s);
            core[size * (i + radius) + j + radius] = res;
            sum += res;
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            core[size * i + j] /= sum;
        }
    }

    return core;
}

vec getRandomMatrix(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Zero or negative arguments passed");
    }

    const int size = width * height;
    std::random_device device;
    std::mt19937 gen(device());
    vec matrix(size);
    for (int i = 0; i < size; i++) {
        matrix[i] = gen() % 100;
    }

    return matrix;
}
