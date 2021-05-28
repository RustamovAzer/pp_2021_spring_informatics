// Copyright 2021 Tyurmina Alexandra
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "../../../modules/task_1/tyurmina_a_shell_betcher/shell_betcher.h"

std::vector<int> ShellSort(const std::vector<int>& massiv, int size) {
    int step;
    int i, j;
    int temp;

    std::vector<int>mass(massiv);

    for (step = size / 2; step > 0; step = step / 2) {
        for (i = step; i < size; i++) {
            for (j = i - step; j >= 0 && mass[j] > mass[j + step]; j = j - step) {
                temp = mass[j];
                mass[j] = mass[j + step];
                mass[j + step] = temp;
            }
        }
    }
    return mass;
}

std::vector<int> BetcherEven(const std::vector<int>& massiv1, const std::vector<int>& massiv2) {
    int size1 = massiv1.size();
    int i1 = 0;
    int size2 = massiv2.size();
    int i2 = 0;
    int i = 0;
    int size_result = size1 / 2 + size2 / 2 + size1 % 2 + size2 % 2;
    std::vector <int> mass_result(size_result);

    while ((i1 < size1) && (i2 < size2)) {
        if (massiv1[i1] <= massiv2[i2]) {
            mass_result[i] = massiv1[i1];
            i1 = i1 + 2;
        } else {
            mass_result[i] = massiv2[i2];
            i2 = i2 + 2;
        }
        i = i + 1;
    }
    if (i1 >= size1) {
        for (int j = i2; j < size2; j = j + 2) {
            mass_result[i] = massiv2[j];
            i = i + 1;
        }
    }
    if (i2 >= size2) {
        for (int j = i1; j < size1; j = j + 2) {
            mass_result[i] = massiv1[j];
            i = i + 1;
        }
    }

    return mass_result;
}

std::vector<int> BetcherOdd(const std::vector<int>& massiv1, const std::vector<int>& massiv2) {
    int size1 = massiv1.size();
    int i1 = 1;
    int size2 = massiv2.size();
    int i2 = 1;
    int i = 0;
    int size_result = size1 / 2 + size2 / 2;
    std::vector <int> mass_result(size_result);

    while ((i1 < size1) && (i2 < size2)) {
        if (massiv1[i1] <= massiv2[i2]) {
            mass_result[i] = massiv1[i1];
            i1 = i1 + 2;
        } else {
            mass_result[i] = massiv2[i2];
            i2 = i2 + 2;
        }
        i = i + 1;
    }
    if (i1 >= size1) {
        for (int j = i2; j < size2; j = j + 2) {
            mass_result[i] = massiv2[j];
            i = i + 1;
        }
    }
    if (i2 >= size2) {
        for (int j = i1; j < size1; j = j + 2) {
            mass_result[i] = massiv1[j];
            i = i + 1;
        }
    }
    return mass_result;
}

std::vector<int> BetcherMerge(const std::vector<int>& massiv1, const std::vector<int>& massiv2) {
    int size1 = massiv1.size();
    int i1 = 0;
    int size2 = massiv2.size();
    int i2 = 0;
    int i = 0;
    int temp;
    int size_result = massiv1.size() + massiv2.size();
    std::vector <int> mass_result(size_result);

    while ((i1 < size1) && (i2 < size2)) {
        mass_result[i] = massiv1[i1];
        mass_result[i + 1] = massiv2[i2];
        i1 = i1 + 1;
        i2 = i2 + 1;
        i = i + 2;
    }
    if ((i1 < size1) && (i2 >= size2)) {
        for (int a = i; a < size_result; a++) {
            mass_result[a] = massiv1[i1];
            i1 = i1 + 1;
        }
    } else {
        for (int a = i; a < size_result; a++) {
            mass_result[a] = massiv1[i2];
            i2 = i2 + 1;
        }
    }
    for (int i = 1; i < size_result; i++) {
        if (mass_result[i] < mass_result[i - 1]) {
            temp = mass_result[i];
            mass_result[i] = mass_result[i - 1];
            mass_result[i - 1] = temp;
        }
    }
    return mass_result;
}

std::vector<int> ShellBetcherS(const std::vector<int>& mass, int size) {
    int p = size / 2;
    std::vector<int> mass1;
    std::vector<int> mass2;
    std::copy(mass.begin(), mass.end() - p, inserter(mass1, mass1.begin()));
    mass1 = ShellSort(mass1, mass1.size());
    std::copy(mass.end() - p, mass.end(), inserter(mass2, mass2.begin()));
    mass2 = ShellSort(mass2, mass2.size());

    std::vector<int>even = BetcherEven(mass1, mass2);
    std::vector<int>odd = BetcherOdd(mass1, mass2);
    std::vector<int>result = BetcherMerge(even, odd);
    return result;
}
