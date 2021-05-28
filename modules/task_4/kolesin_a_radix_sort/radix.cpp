// Copyright 2021 Kolesin Andrey

#include "../../../modules/task_4/kolesin_a_radix_sort/radix.h"

const int base = 256;
const int shift = 8;
const int digits = 4;
const unsigned int masks[digits] = {255u, 255u << 8, 255u << 16, 255u << 24};

void printVec(std::vector<int> vec) {
    if (vec.empty()) {
        return;
    }
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

unsigned int getDigit(int num, int p) {
    unsigned int res = num & masks[p];
    return res >> (shift * p);
}

void radix(int* arr, int size, int p) {
    std::deque<int> digit[base];
    for (int i = 0; i < size; i++) {
        digit[getDigit(arr[i], p)].push_back(arr[i]);
    }

    // for(int i=0;i<base;i++){
    //     std::cout<<"#"<<i<<"! ";
    //     printVec(digit[i]);
    // }
    int pos = 0;
    for (int d = 0; d < base; d++) {
        while (!digit[d].empty()) {
            arr[pos] = digit[d].front();
            digit[d].pop_front();
            pos++;
        }
    }
}

void radixSort(int* arr, int size) {
    for (int i = 0; i < digits; i++) {
        radix(arr, size, i);
    }
    int firstNeg = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0) {
            firstNeg = i;
            break;
        }
    }
    std::rotate(&arr[0], &arr[firstNeg], &arr[size]);
}

std::vector<int> getRandomVector(int size, int maxEl) {
    std::random_device dev;
    std::mt19937 gen(dev());
    if (size == -1) {
        size = 1 + gen() % 20;
    }
    if (maxEl == -1) {
        maxEl = 1 + gen() % 4000000;
    }
    std::vector<int> res;
    for (int i = 0; i < size; i++) {
        res.push_back(gen() % maxEl - maxEl / 2);
    }
    return res;
}

void Merge(int** arrs, int* sizes, int th, int* output) {
    int* pointers = new int[th];
    for (int i = 0; i < th; i++) {
        pointers[i] = 0;
    }
    int s = 0;
    while (true) {
        int min = INT32_MAX;
        int mini = -1;
        for (int id = 0; id < th; id++) {
            if (pointers[id] == sizes[id]) {
                continue;
            }
            if (min > arrs[id][pointers[id]]) {
                min = arrs[id][pointers[id]];
                mini = id;
            }
        }
        if (mini == -1) {
            delete[] pointers;
            return;
        }
        output[s] = min;
        // cout<<min<<" "<<mini<<endl;
        s += 1;
        pointers[mini]++;
    }
}

void parallelTask(int* begins, int* sizes, int id, int** arrs, int* arr) {
    for (int i = begins[id], j = 0; i < begins[id] + sizes[id]; i++, j++) {
        arrs[id][j] = arr[i];
    }
    radixSort(arrs[id], sizes[id]);
}

void parallelSort(int* arr, int size, int threads) {
    int** arrs = new int*[threads];
    for (int i = 0; i < threads; i++) {
        arrs[i] = new int[size];
    }
    int* sizes = new int[threads];
    int* begins = new int[threads];

    for (int i = 0; i < threads; i++) {
        sizes[i] = size / threads;
    }
    if (sizes[0] * threads != size) {
        sizes[0] += size - sizes[0] * threads;
    }
    int b = 0;
    for (int i = 0; i < threads; i++) {
        begins[i] = b;
        b += sizes[i];
    }
    std::thread* thrs = new std::thread[threads];
    for (int id = 0; id < threads; id++) {
        thrs[id] = std::thread(parallelTask, begins, sizes, id, arrs, arr);
    }

    for (int id = 0; id < threads; id++) {
        thrs[id].join();
    }
    Merge(arrs, sizes, threads, arr);
    for (int i = 0; i < threads; i++) {
        delete[] arrs[i];
    }
    delete[] thrs;
    delete[] arrs;
    delete[] sizes;
    delete[] begins;
}
