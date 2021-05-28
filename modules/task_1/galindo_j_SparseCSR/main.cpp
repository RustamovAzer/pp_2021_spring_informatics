// Copyright 2021 Galindo Buitrago Julio Cesar
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./csr_dispersa.h"

TEST(csr_dispersa_test, testeo_de_generacion) {
    int n = 6;
    std::vector<std::complex<double>> matricial = {
        {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 2}, {0, 0},
        {0, 0}, {0, 0}, {0, 3}, {0, 4}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 8}, {0, 0}, {0, 5},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 7}, {0, 1}, {0, 0}, {0, 0}, {0, 6}
    };
    DispersaMatrix M_Dispersa;
    M_Dispersa = ma_generacion(matricial, n);
    std::vector<std::complex<double>> valor_estimado = {
        {0, 1}, {0, 2}, {0, 3},
        {0, 4}, {0, 8}, {0, 5},
        {0, 7}, {0, 1}, {0, 6}
        };
    std::vector<int> columna_estimada = { 0, 4, 2, 3, 3, 5, 1, 2, 5};
    std::vector<int> RowIndex = { 0, 2, 4, 4, 6, 6, 9};
    ASSERT_EQ(M_Dispersa.n, n);
    ASSERT_EQ(M_Dispersa.valor, valor_estimado);
    ASSERT_EQ(M_Dispersa.columnaN, columna_estimada);
    ASSERT_EQ(M_Dispersa.filaN, RowIndex);
}

TEST(csr_dispersa_test, testeo_de_transposicion) {
    int n = 5;
    std::vector<std::complex<double>> matricial = {
        {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 3}, {0, 5}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 0},
        {0, 0}, {0, 2}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 4}, {0, 0}, {0, 2}, {0, 0},
    };
    DispersaMatrix TransponationMat, M_Dispersa;
    M_Dispersa = ma_generacion(matricial, n);
    TransponationMat = transposicion(M_Dispersa);
    std::vector<int> RowIndex = { 0, 2, 5, 5, 7, 7 };
    ASSERT_EQ(TransponationMat.filaN, RowIndex);
}

TEST(csr_dispersa_test, testeo_de_transposicion_dos) {
    int n = 4;
    std::vector<std::complex<double>> matricial = {
        {0, 0}, {0, 2}, {0, 0}, {0, 0},
        {0, 1}, {0, 0}, {0, 0}, {0, 3},
        {0, 0}, {0, 1}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}
    };
    DispersaMatrix TransponationMat, M_Dispersa;
    M_Dispersa = ma_generacion(matricial, n);
    TransponationMat = transposicion(M_Dispersa);
    std::vector<int> RowIndex = { 0, 1, 3, 3, 4 };
    ASSERT_EQ(TransponationMat.filaN, RowIndex);
}

TEST(csr_dispersa_test, testeo_de_multiplicacion) {
    int n = 2;
    std::vector<std::complex<double>> A1 = {
        {0, 0}, {2, 0},
        {1, 0}, {0, 0}
    };
    std::vector<std::complex<double>> A2 = {
        {0, 0}, {2, 0},
        {2, 0}, {0, 0}
    };

    DispersaMatrix Primera_matrix, Segunda_matrix, Solucion;
    Primera_matrix = ma_generacion(A1, n);
    Segunda_matrix = ma_generacion(A2, n);
    Solucion = multiplicacion(Primera_matrix, Segunda_matrix);
    std::vector<std::complex<double>> valor_estimado = { {4, 0}, {2, 0} };
    ASSERT_EQ(Solucion.valor, valor_estimado);
}

TEST(csr_dispersa_test, testeo_de_multiplicacion_dos) {
    int n = 2;
    std::vector<std::complex<double>> A1 = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
    };
    std::vector<std::complex<double>> A2 = {
        {0, 0}, {2, 0},
        {2, 0}, {0, 0}
    };

    DispersaMatrix Primera_matrix, Segunda_matrix, Solucion;
    Primera_matrix = ma_generacion(A1, n);
    Segunda_matrix = ma_generacion(A2, n);
    Solucion = multiplicacion(Primera_matrix, Segunda_matrix);
    std::vector<std::complex<double>> valor_estimado = {};
    ASSERT_EQ(Solucion.valor, valor_estimado);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
