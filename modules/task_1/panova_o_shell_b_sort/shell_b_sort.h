// Copyright 2021 Panova Olga
#ifndef MODULES_TASK_1_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
#define MODULES_TASK_1_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
#include <iostream>
#include <vector>
#include <random>
std::vector<int> CreateVector(int size);
std::vector<int> EvenMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> OddMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> BatcherMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> ShellSort(const std::vector<int>& a);
std::vector<int> OddEvenShellSort(const std::vector<int>& a);
#endif  // MODULES_TASK_1_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
