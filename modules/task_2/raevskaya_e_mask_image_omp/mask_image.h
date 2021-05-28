// Copyright 2021 Ekaterina Raevskaya
#ifndef MODULES_TASK_2_RAEVSKAYA_E_MASK_IMAGE_OMP_MASK_IMAGE_H_
#define MODULES_TASK_2_RAEVSKAYA_E_MASK_IMAGE_OMP_MASK_IMAGE_H_

#include <vector>
#include <utility>

std::pair<std::vector<int>, int> mask_dsu_omp(std::vector<int>* image, int n, int m);
std::vector<int> mask_dsu(std::vector<int>* image, int n, int m);
void make_set(int v);
int find_set(int v);
void union_sets(int a, int b);

#endif  // MODULES_TASK_2_RAEVSKAYA_E_MASK_IMAGE_OMP_MASK_IMAGE_H_
