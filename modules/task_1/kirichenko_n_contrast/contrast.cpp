// Copyright 2021 Kirichenko Nikita
#include "../../../modules/task_1/kirichenko_n_contrast/contrast.h"

Result RandomM(int l, int k) {
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
    int max = 0, min = 255;
    for (int i = 0; i < static_cast<int>(rm.size()); i++) {
        if (min > rm[i])
            min = rm[i];
        if (max < rm[i])
            max = rm[i];
    }
     for (int i = 0; i < static_cast<int>(rm.size()); i++) {
         outcome[i] = (rm[i] - min) * (255 / (max - min));
     }
     return outcome;
}
