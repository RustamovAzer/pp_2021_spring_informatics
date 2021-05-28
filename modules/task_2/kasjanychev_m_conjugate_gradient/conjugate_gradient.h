// Copyright 2021 Kasjanychev Mikhail
#ifndef MODULES_TASK_2_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_
#define MODULES_TASK_2_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_

#include <omp.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

std::vector<std::vector<double> > getRandomMatrix(int);
std::vector<double> getRandomVector(int);
std::vector<double> multiMtxVec(std::vector<std::vector<double> >, std::vector<double>);
std::vector<double> subVec(std::vector<double>, std::vector<double>);
std::vector<double> sumVec(std::vector<double>, std::vector<double>);
double scalarProduct(std::vector<double>, std::vector<double>);
std::vector<double> multiVec(double, std::vector<double>);
std::vector<double> minusVec(std::vector<double>);
std::vector<std::vector<double> > transposeMtx(std::vector<std::vector<double> >);
std::vector<std::vector<double> > sumMtx(std::vector<std::vector<double> >, std::vector<std::vector<double> >);
std::vector<std::vector<double> > multiMtx(std::vector<std::vector<double> >, double);
std::vector<std::vector<double> > searchReverseMatrix(std::vector<std::vector<double> >);
std::vector<double> calculateStandardRes(const std::vector<std::vector<double> >&, std::vector<double>, int);
std::vector<double> calculateRes(std::vector<std::vector<double> >, const std::vector<double>&, int);

#endif  // MODULES_TASK_2_KASJANYCHEV_M_CONJUGATE_GRADIENT_CONJUGATE_GRADIENT_H_
