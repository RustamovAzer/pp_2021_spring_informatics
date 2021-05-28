// Copyright 2021 Galindo Buitrago Julio Cesar
#ifndef MODULES_TASK_2_GALINDO_J_SPARSECSRCONOMP_CSROMP_H_
#define MODULES_TASK_2_GALINDO_J_SPARSECSRCONOMP_CSROMP_H_
#include <omp.h>
#include <vector>
#include <complex>

struct DispersaMatrix {
    int n;
    std::vector<std::complex<double>> valor;
    std::vector<int> columnaN, filaN;
};

DispersaMatrix transposicion(DispersaMatrix AT);
DispersaMatrix multiplicacion(DispersaMatrix p, DispersaMatrix q);
DispersaMatrix multiplicacion_omp(DispersaMatrix p, DispersaMatrix q);
DispersaMatrix ma_generacion(const std::vector<std::complex<double>> &vectorial, int nn);
DispersaMatrix ma_generacion_dia(int nn);

#endif  // MODULES_TASK_2_GALINDO_J_SPARSECSRCONOMP_CSROMP_H_
