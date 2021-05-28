// Copyright 2021 Kiseleva Anastasia
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "./umnrazr.h"

TEST(CCR_UMN, razm3x3_1_1) {
    int nz = 1;
    int n = 3;
    ComplexMatr a = rand(n, n, nz);
    ComplexMatr b = rand(n, n, nz);
    ComplexMatr CCR = umn_posled(a, b, n, n, n, n);
    ComplexMatr umn = proverka(a, b, n, n, n, n);
    for (int i = 0; i < n*n; i++) {
            ASSERT_EQ(CCR[i], umn[i]);
    }
}

TEST(CCR_UMN, razm13x8_8x21_4_8) {
    int na = 13;
    int ma = 8;
    int nb = 8;
    int mb = 21;
    int nza = 4;
    int nzb = 8;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    ComplexMatr CCR = umn_posled(a, b, na, ma, nb, mb);
    ComplexMatr umn = proverka(a, b, na, ma, nb, mb);
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
}

TEST(CCR_UMN, razm113x85_85x21_7_1) {
    int na = 113;
    int ma = 85;
    int nb = 85;
    int mb = 21;
    int nza = 7;
    int nzb = 1;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    ComplexMatr CCR = umn_posled(a, b, na, ma, nb, mb);
    ComplexMatr umn = proverka(a, b, na, ma, nb, mb);
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
}

TEST(CCR_UMN, razm1063x899_899x215_100_26) {
    int na = 1063;
    int ma = 899;
    int nb = 899;
    int mb = 215;
    int nza = 100;
    int nzb = 26;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    ComplexMatr CCR = umn_posled(a, b, na, ma, nb, mb);
    ComplexMatr umn = proverka(a, b, na, ma, nb, mb);
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
}

TEST(CCR_UMN, razm100x500_500x200_2_0) {
    int na = 100;
    int ma = 500;
    int nb = 500;
    int mb = 200;
    int nza = 2;
    int nzb = 0;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    ComplexMatr CCR = umn_posled(a, b, na, ma, nb, mb);
    ComplexMatr umn = proverka(a, b, na, ma, nb, mb);
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
}

TEST(CCR_UMN, razm1000x500_500x200_0_0) {
    int na = 1000;
    int ma = 500;
    int nb = 500;
    int mb = 20;
    int nza = 0;
    int nzb = 0;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    ComplexMatr CCR = umn_posled(a, b, na, ma, nb, mb);
    ComplexMatr umn = proverka(a, b, na, ma, nb, mb);
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
}
