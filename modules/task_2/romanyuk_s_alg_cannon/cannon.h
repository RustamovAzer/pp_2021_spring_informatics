// Copyright 2021 Romanyuk Sergey
#pragma once
#ifndef MODULES_TASK_2_ROMANUYK_S_ALG_CANNON
#define MODULES_TASK_2_ROMANUYK_S_ALG_CANNON

#include <iostream>
#include <vector>
std::vector<double> genMatrix(int n);
std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A,
        const std::vector<double>& B, int n);
std::vector<double> CannonMultiplication(std::vector<double> A,
        std::vector<double> B, int BlockSize);
std::vector<double> parallelCannonMult(const std::vector<double>& A,
        const std::vector<double>& B, int Size);
bool Difference(const double &x, const double &y);
bool DifferenceMatrix(const std::vector<double>&A, const std::vector<double>&B);
#endif  // MODULES_TASK_2_ROMANUYK_S_ALG_CANNON
