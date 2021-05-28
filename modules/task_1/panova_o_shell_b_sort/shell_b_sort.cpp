// Copyright 2021 Panova Olga
#include "../../../modules/task_1/panova_o_shell_b_sort/shell_b_sort.h"
std::vector<int> CreateVector(int size) {
    std::random_device d;
    std::mt19937 g(d());
    std::vector<int> a(size);
    for (int i = 0; i < size; i++) {
        a[i] = g() % 111;
    }
    return a;
}
std::vector<int> EvenMerge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> res;
    int left_size = static_cast<int>(left.size());
    int right_size = static_cast<int>(right.size());
    int left_iter = 0, right_iter = 0;
    res.reserve((left_size + 1) / 2 + (right_size + 1) / 2);
    while ((left_iter < left_size) && (right_iter < right_size)) {
        if (left[left_iter] <= right[right_iter]) {
            res.push_back(left[left_iter]);
            left_iter += 2;
        } else {
            res.push_back(right[right_iter]);
            right_iter += 2;
        }
    }
    if (right_iter >= right_size) {
        for (int i = left_iter; i < left_size; i += 2) {
            res.push_back(left[i]);
        }
    } else {
        for (int i = right_iter; i < right_size; i += 2) {
            res.push_back(right[i]);
        }
    }
    return res;
}
std::vector<int> OddMerge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> res;
    int left_size = static_cast<int>(left.size());
    int right_size = static_cast<int>(right.size());
    int left_iter = 1, right_iter = 1;
    res.reserve(left_size / 2 + right_size / 2);
    while ((left_iter < left_size) && (right_iter < right_size)) {
        if (left[left_iter] <= right[right_iter]) {
            res.push_back(left[left_iter]);
            left_iter += 2;
        } else {
            res.push_back(right[right_iter]);
            right_iter += 2;
        }
    }
    if (right_iter >= right_size) {
        for (int i = left_iter; i < left_size; i += 2) {
            res.push_back(left[i]);
        }
    } else {
        for (int i = right_iter; i < right_size; i += 2) {
            res.push_back(right[i]);
        }
    }
    return res;
}
std::vector<int> BatcherMerge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> res;
    int left_size = static_cast<int>(left.size());
    int right_size = static_cast<int>(right.size());
    res.reserve(left_size + right_size);
    if (left_size == right_size) {
        for (int i = 0; i < left_size; i++) {
            res.push_back(left[i]);
            res.push_back(right[i]);
        }
    } else if (left_size == right_size + 2 || right_size == left_size + 2) {
        int i = 0;
        for (i = 0; i < std::min(left_size, right_size); i++) {
            res.push_back(left[i]);
            res.push_back(right[i]);
        }
        if (left_size == right_size + 2) {
            res.push_back(left[i]);
            res.push_back(left[i + 1]);
        } else {
            res.push_back(right[i]);
            res.push_back(right[i + 1]);
        }
    }
    for (int i = 1; i < left_size + right_size; i++) {
        if (res[i] < res[i - 1]) {
            std::swap(res[i], res[i - 1]);
        }
    }
    return res;
}
std::vector<int> ShellSort(const std::vector<int>& a) {
    std::vector<int> res(a);
    int size = static_cast<int>(res.size());
    int k, incr, tmp;
    int swp = 0, comp = 0; incr = size / 2;
    while (incr > 0) {
        for (int i = 0; i < incr; i++) {
            for (int j = 0; j < size; j += incr) {
                tmp = res[j];
                for (k = j - incr; k >= 0 && tmp < res[k]; k -= incr) {
                    comp++;
                    swp++;
                    res[k + incr] = res[k];
                }
                res[k + incr] = tmp;
                swp++;
            }
        }
        comp++;
        if (incr / 2 != 0) {
            incr = incr / 2;
        } else if (incr == 1) {
            incr = 0;
        } else {
            incr = 1;
        }
    }
    return res;
}
std::vector<int> OddEvenShellSort(const std::vector<int>& a) {
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> even;
    std::vector<int> odd;
    int size = static_cast<int>(a.size());
    if (size <= 1) {
        return a;
    }
    int half = (size + 1) / 2;
    left.reserve(half);
    right.reserve(half);
    for (int i = 0; i < half; i++) {
        left.push_back(a[i]);
    }
    for (int i = half; i < size; i++) {
        right.push_back(a[i]);
    }
    left = ShellSort(left);
    right = ShellSort(right);
    even = EvenMerge(left, right);
    odd = OddMerge(left, right);
    return BatcherMerge(even, odd);
}
