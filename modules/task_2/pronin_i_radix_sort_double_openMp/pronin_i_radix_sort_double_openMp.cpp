// Copyright 2021 Pronin Igor
#include "../../../modules/task_2/pronin_i_radix_sort_double_openMp/pronin_i_radix_sort_double_openMp.h"
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

void Merge(double* mas1, double* mas2, double* mas3, int size1, int size2) {
    int m1 = 0;
    int m2 = 0;
    int m3 = 0;
    while (m1 < size1 && m2 < size2) {
        if (mas1[m1] < mas2[m2]) {
            mas3[m3] = mas1[m1];
            m1++;
        } else {
            mas3[m3] = mas2[m2];
            m2++;
        }
        m3++;
    }
    while (m1 < size1) {
        mas3[m3] = mas1[m1];
        m1++;
        m3++;
    }
    while (m2 < size2) {
        mas3[m3] = mas2[m2];
        m2++;
        m3++;
    }
}

void SeqSort(double* inp, double* out, int size, int threads) {
    int length = size / threads;
    std::vector<int> len(threads);
    for (int i = 0; i < threads; i++)
        len[i] = length;
    len[threads - 1] += size % threads;
    std::vector<int> offset(threads);
    offset[0] = 0;
    for (int i = 1; i < threads; i++)
        offset[i] = offset[i - 1] + len[i - 1];
    for (int i = 0; i < threads; i++)
        SortAll(inp + offset[i], out, len[i]);
    int povtor = threads / 2 + threads % 2;
    int num = 2;
    int step = 1;
    for (int j = 0; j < povtor; j++) {
        for (int i = 0; i < threads; i++)
            if (i % num == 0 && i + step < threads) {
                double* part = new double[size];
                Merge(inp + offset[i], inp + offset[i + step], part, len[i],
                    len[i + step]);
                int len12 = len[i] + len[i + step];
                int icopy = offset[i];
                for (int k = 0; k < len12; k++) {
                    inp[icopy] = part[k];
                    icopy++;
                }
                len[i] += len[i + step];
                delete[] part;
            }
        num *= 2;
        step *= 2;
    }
    for (int i = 0; i < size; i++)
        out[i] = inp[i];
}

void ParSort(double* inp, double* out, int size, int threads) {
    int length = size / threads;
    std::vector<int> len(threads);
    for (int i = 0; i < threads; i++)
        len[i] = length;
    len[threads - 1] += size % threads;
    std::vector<int> offset(threads);
    offset[0] = 0;
    for (int i = 1; i < threads; i++)
        offset[i] = offset[i - 1] + len[i - 1];
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < threads; i++) {
        double* in = new double[size];
        SortAll(inp + offset[i], in, len[i]);
    }
    int povtor = threads / 2 + threads % 2;
    int num = 2;
    int step = 1;
    for (int j = 0; j < povtor; j++) {
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < threads; i++)
            if (i % num == 0 && i + step < threads) {
                double* part = new double[size];
                Merge(inp + offset[i], inp + offset[i + step], part, len[i],
                    len[i + step]);
                int len12 = len[i] + len[i + step];
                int icopy = offset[i];
                for (int k = 0; k < len12; k++) {
                    inp[icopy] = part[k];
                    icopy++;
                }
                len[i] += len[i + step];
                delete[] part;
            }
        num *= 2;
        step *= 2;
    }
    for (int i = 0; i < size; i++)
        out[i] = inp[i];
}
