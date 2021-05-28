// Copyright 2021 Belik Julia
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>
#include "../../../modules/task_1/belik_j_radix_sort/RadixSortB.h"
std::vector<double> RadixSort(std::vector<double> vec) {
    std::vector<double> res = vec;
    for (size_t i = 0; i < 7; i++) {
        unsigned char* bvec = reinterpret_cast<unsigned char*>(res.data());
        int countb[256] = { 0 };
        int offset[256] = { 0 };
        std::vector<double> res1(vec.size());
        for (size_t j = 0; j < vec.size(); j++)
            countb[bvec[8 * j + i]]++;
        for (int j = 1; j < 256; j++)
            offset[j] = offset[j - 1] + countb[j - 1];
        for (size_t j = 0; j < res.size(); j++)
            res1[offset[bvec[8 * j + i]]++] = res[j];
        for (size_t j = 0; j < res.size(); j++)
            res[j] = res1[j];
    }
    unsigned char* bvec = reinterpret_cast<unsigned char*>(res.data());
    int countb[256] = { 0 };
    int offset[256] = { 0 };
    std::vector<double> res1(res.size());
    for (size_t i = 0; i < res.size(); i++)
        countb[bvec[8 * i + 7]]++;
    offset[255] = countb[255] - 1;
    for (int i = 254; i >= 128; i--)
        offset[i] = offset[i + 1] + countb[i];
    offset[0] = offset[128] + 1;
    for (int i = 1; i < 128; i++)
        offset[i] = offset[i - 1] + countb[i - 1];
    for (size_t i = 0; i < vec.size(); i++) {
        if (bvec[8 * i + 7] >= 128) {
            res1[offset[bvec[8 * i + 7]]--] = res[i];
        } else {
            res1[offset[bvec[8 * i + 7]]++] = res[i];
        }
    }
    return res1;
}
std::vector<double> Vector(int n, double a, double b) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<double> rand(a, b);
    std::vector<double> vec(n);
    for (int i = 0; i < n; i++)
        vec[i] = rand(gen);
    return vec;
}
