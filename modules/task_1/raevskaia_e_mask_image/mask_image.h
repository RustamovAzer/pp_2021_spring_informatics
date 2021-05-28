// Copyright 2021 Ekaterina Raevskaya
#ifndef MODULES_TASK_1_RAEVSKAIA_E_MASK_IMAGE_MASK_IMAGE_H_
#define MODULES_TASK_1_RAEVSKAIA_E_MASK_IMAGE_MASK_IMAGE_H_

#include <vector>

bool valid(int x, int y, int n, int m);
std::vector<int> mask(std::vector<int>* image, int n, int m);
void dfs(std::vector<int>* image, int x, int y, int n, int m,
         int value, std::vector<int>* res_image);

#endif  // MODULES_TASK_1_RAEVSKAIA_E_MASK_IMAGE_MASK_IMAGE_H_
