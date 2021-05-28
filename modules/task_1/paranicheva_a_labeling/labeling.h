// Copyright 2021 Paranicheva Alyona
#ifndef MODULES_TASK_1_PARANICHEVA_A_LABELING_LABELING_H_
#define MODULES_TASK_1_PARANICHEVA_A_LABELING_LABELING_H_

#include <vector>
#include <utility>

std::vector<int> getRandomMatrix(int rows, int cols);
std::pair<std::vector<int>, std::vector<int> > firstMark(std::vector<int>, int, int);
std::vector<int> secondMark(std::vector<int>, int, int, std::vector<int>);
std::vector<int> CLabeling(const std::vector<int>&, int, int);

#endif  // MODULES_TASK_1_PARANICHEVA_A_LABELING_LABELING_H_
