// Copyright 2021 Emelkhovskaya Ekaterina

#include <math.h>
#include <string>
#include <stdexcept>
#include "../../modules/task_2/emelkhovskaya_e_strassen_matrix_mult_omp/strassen.h"

void seqMult(double* a, double* b, double* c, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }
}

void P1(double* a11, double* a22, double* b11,
double* b22, double* prores1, int size) {
    double* res1 = new double[size * size * 2];
    double* res2 = res1 + size * size;
    for (int i = 0; i < size * size; i++) {
        res1[i] = a11[i] + a22[i];
        res2[i] = b11[i] + b22[i];
    }
    StrassenFunc(res1, res2, prores1, size);
    delete[] res1;
}
void P2(double* a21, double* a22, double* b11,
double* prores2, int size) {
    double* res1 = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        res1[i] = a21[i] + a22[i];
    }
    StrassenFunc(res1, b11, prores2, size);
    delete[] res1;
}
void P3(double* a11, double* b12, double* b22,
double* prores3, int size) {
    double* res1 = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        res1[i] = b12[i] - b22[i];
    }
    StrassenFunc(a11, res1, prores3, size);
    delete[] res1;
}
void P4(double* a22, double* b21, double* b11,
double* prores4, int size) {
    double* res1 = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        res1[i] = b21[i] - b11[i];
    }
    StrassenFunc(a22, res1, prores4, size);
    delete[] res1;
}
void P5(double* a11, double* a12, double* b22,
double* prores5, int size) {
    double* res1 = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        res1[i] = a11[i] + a12[i];
    }
    StrassenFunc(res1, b22, prores5, size);
    delete[] res1;
}
void P6(double* a21, double* a11, double* b11,
double* b12, double* prores6, int size) {
    double* res1 = new double[size * size * 2];
    double* res2 = res1 + size * size;
    for (int i = 0; i < size * size; i++) {
        res1[i] = a21[i] - a11[i];
        res2[i] = b11[i] + b12[i];
    }
    StrassenFunc(res1, res2, prores6, size);
    delete[] res1;
}
void P7(double* a12, double* a22, double* b21,
double* b22, double* prores7, int size) {
    double* res1 = new double[size * size * 2];
    double* res2 = res1 + size * size;
    for (int i = 0; i < size * size; i++) {
        res1[i] = a12[i] - a22[i];
        res2[i] = b21[i] + b22[i];
    }
    StrassenFunc(res1, res2, prores7, size);
    delete[] res1;
}


void StrassenFunc(double* a, double* b, double* c, int size) {
    bool flag2 = true;

    for (int i = 0; i < size; i++) {
        if (pow(2, i) == size) {
            break;
        }
        if (pow(2, i) > size) {
            flag2 = false;
            break;
        }
    }
    if (!flag2) {
        throw std::invalid_argument("no power 2");
    }

    if (size <= 32) {
        seqMult(a, b, c, size);
        return;
    }

    int part_size = static_cast<int>(size * size / 4);

    double* a11 = new double[size * size * 2];
    double* a12 = a11 + part_size;
    double* a21 = a12 + part_size;
    double* a22 = a21 + part_size;

    double* b11 = a22 + part_size;
    double* b12 = b11 + part_size;
    double* b21 = b12 + part_size;
    double* b22 = b21 + part_size;

    double* c11 = new double[size * size + part_size * 9];
    double* c12 = c11 + part_size;
    double* c21 = c12 + part_size;
    double* c22 = c21 + part_size;

    double* prores1 = c22 + part_size;
    double* prores2 = prores1 + part_size;
    double* prores3 = prores2 + part_size;
    double* prores4 = prores3 + part_size;
    double* prores5 = prores4 + part_size;
    double* prores6 = prores5 + part_size;
    double* prores7 = prores6 + part_size;

    int hauf = static_cast<int>(size / 2);
    for (int i = 0; i < hauf; i++) {
        for (int j = 0; j < hauf; j++) {
            a11[i * hauf + j] = a[i * size + j];
            a12[i * hauf + j] = a[i * size + hauf + j];
            a21[i * hauf + j] = a[(hauf + i) * size + j];
            a22[i * hauf + j] = a[(hauf + i) * size + hauf + j];

            b11[i * hauf + j] = b[i * size + j];
            b12[i * hauf + j] = b[i * size + hauf + j];
            b21[i * hauf + j] = b[(hauf + i) * size + j];
            b22[i * hauf + j] = b[(hauf + i) * size + hauf + j];
        }
    }

#pragma omp parallel sections shared(a11, a12, a21, a22, b11, b12, b21, b22)
    {
#pragma omp section
        {P1(a11, a22, b11, b22, prores1, size / 2); }
#pragma omp section
        {P2(a21, a22, b11, prores2, size / 2); }
#pragma omp section
        {P3(a11, b12, b22, prores3, size / 2); }
#pragma omp section
        {P4(a22, b21, b11, prores4, size / 2); }
#pragma omp section
        {P5(a11, a12, b22, prores5, size / 2); }
#pragma omp section
        {P6(a21, a11, b11, b12, prores6, size / 2); }
#pragma omp section
        {P7(a12, a22, b21, b22, prores7, size / 2); }
    }
    for (int i = 0; i < (size / 2) * (size / 2); i++) {
        c11[i] = prores1[i] + prores4[i] - prores5[i] + prores7[i];
        c12[i] = prores3[i] + prores5[i];
        c21[i] = prores2[i] + prores4[i];
        c22[i] = prores1[i] - prores2[i] + prores3[i] + prores6[i];
    }

    for (int i = 0; i < hauf; i++) {
        for (int j = 0; j < hauf; j++) {
            c[i * size + j] = c11[i * hauf + j];
            c[i * size + hauf + j] = c12[i * hauf + j];
            c[(hauf + i) * size + j] = c21[i * hauf + j];
            c[(hauf + i) * size + hauf + j] = c22[i * hauf + j];
        }
    }

    delete[] a11;
    delete[] c11;
}
