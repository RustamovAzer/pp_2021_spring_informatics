// Copyright 2021 Galindo Buitrago Julio Cesar
#include <float.h>
#include <random>
#include <vector>
#include "../../../modules/task_1/galindo_j_SparseCSR/csr_dispersa.h"

DispersaMatrix transposicion(DispersaMatrix AT) {
    DispersaMatrix transposed;
    std::vector<std::vector<std::complex<double>>> AlmacenamientoValor(AT.n);
    std::vector<std::vector<int>> AlmacenamientoColumna(AT.n);
    transposed.filaN.push_back(0);
    for (int i = 0; i < AT.n; i++) {
        for (int j = AT.filaN[i]; j < AT.filaN[i + 1]; j++) {
            AlmacenamientoValor[AT.columnaN[j]].push_back(AT.valor[j]);
            AlmacenamientoColumna[AT.columnaN[j]].push_back(i);
        }
    }
    for (int i = 0; i < AT.n; i++) {
        for (int j = 0; j < static_cast<int>(AlmacenamientoColumna[i].size()); j++) {
            transposed.columnaN.push_back(AlmacenamientoColumna[i][j]);
            transposed.valor.push_back(AlmacenamientoValor[i][j]);
        }
    }
    int adicion = 0;
    for (int i = 0; i < AT.n; i++) {
        adicion += static_cast<int>(AlmacenamientoColumna[i].size());
        transposed.filaN.push_back(adicion);
    }
    return transposed;
}

DispersaMatrix multiplicacion(DispersaMatrix p, DispersaMatrix q) {
    DispersaMatrix r;
    std::vector<std::complex<double>> valor_c;
    std::vector<int> columnaN_c;
    std::vector<int> filaN_c;
    if (p.n != q.n)
        throw(std::string)"incorrect size";
    q = transposicion(q);
    filaN_c.push_back(0);
    for (int fila_en_a = 0; fila_en_a < p.n; fila_en_a++) {
        for (int fila_en_b = 0; fila_en_b < p.n; fila_en_b++) {
            std::complex<double> escalar_s = 0;
            for (int k = p.filaN[fila_en_a]; k < p.filaN[fila_en_a + 1]; k++) {
                for (int n = q.filaN[fila_en_b]; n < q.filaN[fila_en_b + 1]; n++) {
                    if (p.columnaN[k] == q.columnaN[n]) {
                        escalar_s += p.valor[k] * q.valor[n];
                        break;
                    }
                }
            }
            if (std::abs(escalar_s) >= DBL_MIN) {
                valor_c.push_back(escalar_s);
                columnaN_c.push_back(fila_en_b);
            }
        }
        filaN_c.push_back(valor_c.size() + filaN_c[fila_en_a]);
    }
    for (int j = 0; j < static_cast<int>(columnaN_c.size()); j++) {
        r.columnaN.push_back(columnaN_c[j]);
        r.valor.push_back(valor_c[j]);
    }
    for (int i = 0; i <= p.n; i++) {
        r.filaN.push_back(filaN_c[i]);
    }
    return r;
}

DispersaMatrix ma_generacion(const std::vector<std::complex<double>> &vectorial, int m) {
    DispersaMatrix matrix;
    matrix.n = m;
    matrix.filaN.push_back(0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            if (std::abs(vectorial[i*m + j]) < DBL_MIN)
                continue;
            matrix.valor.push_back(vectorial[i*m + j]);
            matrix.columnaN.push_back(j);
        }
        matrix.filaN.push_back(matrix.columnaN.size());
    }
    return matrix;
}
