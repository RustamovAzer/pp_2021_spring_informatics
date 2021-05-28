// Copyright 2021 Luckianchencko Ivan
#ifndef MODULES_TASK_2_LUCKIANCHENCKO_I_OBOLOCHKA_OBOLOCHKA_H_
#define MODULES_TASK_2_LUCKIANCHENCKO_I_OBOLOCHKA_OBOLOCHKA_H_

#include<omp.h>
#include<cmath>
#include<vector>
#include<iostream>
#include<ctime>
#include<random>
#include<algorithm>
using mat = std::vector<std::vector<int>>;
mat get_rand_mat(int size);
struct point {
    int x, y;
};
double distance(point a, point b);
double cosAngle(point a, point b, point c);
std::vector<point> Jarvis_alg(const mat &a, int size);
std::vector<point> get_points(mat img);
std::vector<point> Paral_Jarvis_alg(const mat &img, int size);

#endif  // MODULES_TASK_2_LUCKIANCHENCKO_I_OBOLOCHKA_OBOLOCHKA_H_
