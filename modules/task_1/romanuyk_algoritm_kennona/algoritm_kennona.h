// Copyright 2021 Romanuyk Sergey
#ifndef MODULES_TASK_1_ROMANUYK_ALGORITM_KENNONA_ALGORITM_KENNONA_H_
#define MODULES_TASK_1_ROMANUYK_ALGORITM_KENNONA_ALGORITM_KENNONA_H_

#include <iostream>
#include <vector>
std::vector<double> genMatrix(int n);
std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A,
    const std::vector<double>& B, int n);
std::vector<double> KennonMultiplication(std::vector<double> A,
    std::vector<double> B, int BlockSize);

#endif  // MODULES_TASK_1_ROMANUYK_ALGORITM_KENNONA_ALGORITM_KENNONA_H_
