// Copyright 2021 Streltsova Yana
#ifndef MODULES_TASK_1_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
#define MODULES_TASK_1_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_

#include <vector>
#include <utility>

std::vector<int> getRandomVector(int  sz);

template< typename T>
bool compare(T first, T second) {
    return first < second ? true : false;
}

template< typename RandomAccessIterator, typename T >
void shell_sort(RandomAccessIterator first, RandomAccessIterator last, bool (*Compare)(T, T)) {
    for (auto d = (last - first) / 2; d != 0; d /= 2)
        for (auto i = first + d; i != last; ++i)
            for (auto j = i; j - first >= d && compare<int>(*j, *(j - d)); j -= d)
                std::swap(*j, *(j - d));
}

#endif  // MODULES_TASK_1_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
