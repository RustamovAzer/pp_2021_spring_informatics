// Copyright 2021 Tyurmina Alexandra
#ifndef MODULES_TASK_2_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_
#define MODULES_TASK_2_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_

#include <iostream>
#include <vector>
#include <utility>
#include <functional>

std::vector<int> Random(int size);
std::vector<std::vector<int>> Separat(const std::vector<int>& massiv, size_t n);
std::vector<int> ShellSort(const std::vector<int>&, int size);
std::vector<int> BetcherEven(const std::vector<int>&, const std::vector<int>&);
std::vector<int> BetcherOdd(const std::vector<int>&, const std::vector<int>&);
std::vector<int> BetcherMerge(const std::vector<int>&, const std::vector<int>&);
std::vector<int> ShellBetcher(const std::vector<int>&, int);
std::vector<int> ShellBetcherOmp(const std::vector<int>&, const int, int);
std::vector<int> MergeOmp(const std::vector<std::vector<int>>&, const int, int);

#endif  // MODULES_TASK_2_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_
