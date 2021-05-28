// Copyright 2021 Nikolaev Denis
#pragma once
#ifndef MODULES_TASK_1_NIKOLAEV_D_LINEAL_FILTRATION_GAUSS_VERTIC
#define MODULES_TASK_1_NIKOLAEV_D_LINEAL_FILTRATION_GAUSS_VERTIC

#include <vector>
#include <random>
#include <ctime>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;

int clamp(int n, int upper, int lower);
Matrix createrandmatrix(int i, int j);
std::vector<double> transposition(const std::vector<double>& matrix,
    int i, int j);
void printMatrix(Matrix img, int i, int j);
Matrix gengausskernel(int R, int beta);
Matrix sequentialgaussfilt(Matrix Image, int i, int j,
    int R, double beta);
#endif  // MODULES_TASK_1_NIKOLAEV_D_LINEAL_FILTRATION_GAUSS_VERTIC
