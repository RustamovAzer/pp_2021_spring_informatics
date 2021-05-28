// Copyright 2020 Chistov Vladimir

#include <math.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "../../../modules/task_1/chistov_v_gauss_block/gauss_block.h"

// fix3

std::vector<double> Gauss_Sequential(std::vector<double> image,
    int width, int height) {
    std::vector<double> calc((width + 2) * (height + 2));
    for (int x = 0; x < width + 2; x++) {
        for (int y = 0; y < height + 2; y++) {
            if ((x == 0) || (y == 0) || (x == width + 1) || (y == height + 1)) {
                calc[x * (height + 2) + y] = 0;
            } else {
                calc[x * (height + 2) + y] = image[(x - 1) * height + y - 1];
            }
        }
    }
    int count = 0;
    std::vector<double> res(width * height);
    for (int x = 1; x < width + 1; x++) {
        for (int y = 1; y < height + 1; y++) {
            double sum = 0;
            int gk = 0;
            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    sum = sum + calc[(x + i) * (height
                        + 2) + y + j] * Gauss_Core[gk];
                    gk++;
                }
            }
            res[count] = sum / 16;
            count++;
        }
    }
    return res;
}

std::vector<double> GenRandVec(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = gen() % 256;
    }
    return vec;
}

std::vector <double> Block_Construct(std::vector<double> image,
    int num, int widthloc, int heightloc, int heigth) {
    std::vector<double> calc(widthloc * heightloc);
    std::vector<double> res(widthloc * heightloc);
    int a = 0;
    for (int i = 0; i < widthloc; i++) {
        for (int j = 0; j < heightloc; j++) {
            calc[a] = image[num + i * heigth + j];
            a++;
        }
    }
    a = 0;
    res = Gauss_Sequential(calc, widthloc, heightloc);
    return res;
}

std::vector <double> Block_Destruct(std::vector<double> empty,
    std::vector<double> block, int num,
int widthloc, int heightloc, int heigth) {
    int a = 0;
    for (int i = 0; i < widthloc; i++) {
        for (int j = 0; j < heightloc; j++) {
            empty[num + i * heigth + j] = block[a];
            a++;
        }
    }
    a = 0;
    return empty;
}

std::vector<double> Gauss_Parallel(const std::vector<double> &image,
    int width, int height, int num) {
    int sidenum = ceil(static_cast<double>(sqrt(num)));
    int blockwidth = width / sidenum;
    int blockwidthrem = width % sidenum;
    int blockheight = height / sidenum;
    int blockheightrem = height % sidenum;
    int heightcalc, widthcalc, hrem, wrem;
    int locbh = 0;
    int locbw = 0;
    int start = 0;
    std::vector<double> res(width * height);
    if (blockwidthrem > 0) {
        widthcalc = 1;
    } else {
        widthcalc = 0;
    }
    if (blockheightrem > 0) {
        heightcalc = 1;
    } else {
        heightcalc = 0;
    }
    wrem = blockwidthrem;
    for (int i = 0; i < sidenum; i++) {
        hrem = blockheightrem;
        if (wrem < 1) {
            widthcalc = 0;
        } else {
            wrem--;
            widthcalc = 1;
        }
        locbw = blockwidth + widthcalc;
        for (int j = 0; j < sidenum; j++) {
            if (hrem < 1) {
                heightcalc = 0;
            } else {
                heightcalc = 1;
                hrem--;
            }
            locbh = blockheight + heightcalc;
            // auto block = Block_Construct(image, start, locbw, locbh, height);
            res = Block_Destruct(res, Block_Construct(image, start, locbw,
                locbh, height), start, locbw, locbh, height);
            start = start + locbh;
        }
        start = start + height * (locbw - 1);
    }
    return res;
}

