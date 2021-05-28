// Copyright 2021 Tyurmina Alexandra
#ifndef MODULES_TASK_1_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_
#define MODULES_TASK_1_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_
#include <iostream>
#include <vector>
#include <random>

std::vector<int> ShellSort(const std::vector<int>& massiv, int size);
std::vector<int> ShellBetcherS(const std::vector<int>& mass, int size);
std::vector<int> BetcherMerge(const std::vector<int>& massiv1, const std::vector<int>& massiv2);
std::vector<int> BetcherEven(const std::vector<int>& massiv1, const std::vector<int>& massiv2);
std::vector<int> BetcherOdd(const std::vector<int>& massiv1, const std::vector<int>& massiv2);

#endif  // MODULES_TASK_1_TYURMINA_A_SHELL_BETCHER_SHELL_BETCHER_H_
