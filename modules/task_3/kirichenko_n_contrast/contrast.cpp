// Copyright 2021 Kirichenko Nikita
#include "../../../modules/task_3/kirichenko_n_contrast/contrast.h"

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
        outcome[i] = (rm[i] - min) * (255 / (max - min));
    }
    return outcome;
}

Result Contrasttbb(const Result& rm) {
    Result outcome(rm.size());
    int min = 255, max = 0;

    int const numThreads = 4;

    tbb::task_scheduler_init init(numThreads);

    size_t gs = ((rm.size() / 1000) > 1 ? (rm.size() / 1000) : 1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rm.size(), gs),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (rm[i] > max) {
                    max = rm[i];
                }
                if (rm[i] < min) {
                    min = rm[i];
                }
            }
        }, tbb::simple_partitioner());

    tbb::parallel_for(0, static_cast<int>(rm.size()), [&](const int i) {
        outcome[i] = ((rm[i] - min) * 255) / (max - min);
        });
    return outcome;
}
