// Copyright 2021 Kirichenko Nikita
#include "../../../modules/task_2/kirichenko_n_contrast/contrast.h"

Result RandomI(int l, int k) {
    if (l <= 0 || k <= 0)
        throw std::runtime_error("Incorrect data!");
    static std::mt19937 random(time(0));
    Result res(l * k);
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < k; j++) {
            res[i * l + j] = random() % 256;
        }
    }
    return res;
}

Result Contrast(const Result& rm) {
    Result outcome(rm.size());
    int max = 0;
    int min = 255;
    for (int i = 0; i < static_cast<int>(rm.size()); i++) {
        if (min > rm[i])
            min = rm[i];
        if (max < rm[i])
            max = rm[i];
    }
    for (int i = 0; i < static_cast<int>(rm.size()); i++) {
        outcome[i] = ((rm[i] - min) * 255) / (max - min);
    }
    return outcome;
}

Result Contrastomp(const Result& rm) {
    Result outcome(rm.size());
    int const num_threads = 8;
    omp_set_num_threads(num_threads);
    int max_n[num_threads], min_n[num_threads];
    int max = 0, min = 255;
#pragma omp parallel
    {
       int nT = omp_get_thread_num();
        min_n[nT] = 255;
        max_n[nT] = 0;
#pragma omp for
        for (int i = 0; i < static_cast<int>(rm.size()); i++) {
            if (min_n[nT] > rm[i])
                min_n[nT] = rm[i];
            if (max_n[nT] < rm[i])
                max_n[nT] = rm[i];
        }
    }
#pragma omp for
    for (int i = 0; i < num_threads; i++) {
        if (min_n[i] < min)
            min = min_n[i];
        if (max_n[i] > max)
            max = max_n[i];
    }
#pragma omp for
    for (int i = 0; i < static_cast<int>(rm.size()); i++) {
        outcome[i] = ((rm[i] - min) * 255) / (max - min);
    }
    return outcome;
}
