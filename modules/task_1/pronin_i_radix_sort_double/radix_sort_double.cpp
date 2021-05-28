// Copyright 2021 Pronin Igor
#include "../../../modules/task_1/pronin_i_radix_sort_double/radix_sort_double.h"
void RandVec(double* vec, int left, int right, int size) {
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> urd(left, right);
    for (int i = 0; i < size; i++)
        vec[i] = urd(gen);
}

void SortBite(double* inp, double* out, int byteNum, int size) {
    unsigned char* vec = (unsigned char*)inp;
    int counter[256] = { 0 };
    int offset[256] = { 0 };
    for (int i = 0; i < size; i++)
        counter[vec[i * 8 + byteNum]]++;
    for (int i = 1; i < 256; i++)
        offset[i] = offset[i - 1] + counter[i - 1];
    for (int i = 0; i < size; i++) {
        out[offset[vec[i * 8 + byteNum]]] = inp[i];
        offset[vec[i * 8 + byteNum]]++;
    }
    if (byteNum == 7) {
        int negNum = 0;
        for (int i = 0; i < size; i++)
            inp[i] = out[i];
        for (int i = 0; i < size; i++)
            if (inp[i] < 0)
                negNum++;
        for (int i = 0; i < negNum; i++)
            out[i] = inp[size - i - 1];
        for (int i = negNum; i < size; i++)
            out[i] = inp[i - negNum];
    }
}

void SortAll(double* inp, double* out, int size) {
    SortBite(inp, out, 0, size);
    SortBite(out, inp, 1, size);
    SortBite(inp, out, 2, size);
    SortBite(out, inp, 3, size);
    SortBite(inp, out, 4, size);
    SortBite(out, inp, 5, size);
    SortBite(inp, out, 6, size);
    SortBite(out, inp, 7, size);
}
