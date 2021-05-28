// Copyright 2021 Emelkhovskaya Ekaterina

#ifndef MODULES_TASK_1_EMELKHOVSKAYA_E_STRASSEN_MATRIX_MULT_STRASSEN_H_
#define MODULES_TASK_1_EMELKHOVSKAYA_E_STRASSEN_MATRIX_MULT_STRASSEN_H_

void StrassenFunc(double* a, double* b, double* c, int size);
void seqMult(double* a, double* b, double* c, int size);

void P1(double* a11, double* a22, double* res1, double* b11, double* b22,
    double* res2, double* prores1, int size);
void P2(double* a21, double* a22, double* res1, double* b11, double* prores2,
    int size);
void P3(double* a11, double* res1, double* b12, double* b22, double* prores3,
    int size);
void P4(double* a22, double* res1, double* b21, double* b11, double* prores4,
    int size);
void P5(double* a11, double* a12, double* res1, double* b22, double* prores5,
    int size);
void P6(double* a21, double* a11, double* res1, double* b11, double* b12,
    double* res2, double* prores6, int size);
void P7(double* a12, double* a22, double* res1, double* b21, double* b22,
    double* res2, double* prores7, int size);

#endif  // MODULES_TASK_1_EMELKHOVSKAYA_E_STRASSEN_MATRIX_MULT_STRASSEN_H_
