// Copyright 2021 Kiseleva Anastasia
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <complex>
#include "./umnrazr.h"

TEST(CCR_UMN, razm1020x100_100_10000) {
    int na = 1020;
    int ma = 100;
    int nb = 100;
    int mb = 100;
    int nza = 4;
    int nzb = 8;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
}

TEST(CCR_UMN, razm100x100_100x100_4_8) {
    int na = 100;
    int ma = 100;
    int nb = 100;
    int mb = 100;
    int nza = 4;
    int nzb = 8;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
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
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
}

TEST(CCR_UMN, razm25000x10000_10000x25000_110_80) {
    int na = 250;
    int ma = 100;
    int nb = 100;
    int mb = 250;
    int nza = 110;
    int nzb = 80;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
}

TEST(CCR_UMN, razm10000x10000_10000x10000_10_8) {
    int na = 100;
    int ma = 100;
    int nb = 100;
    int mb = 100;
    int nza = 10;
    int nzb = 8;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
}

TEST(CCR_UMN, razm1000x500_500x200_10_0) {
    int na = 1000;
    int ma = 500;
    int nb = 500;
    int mb = 20;
    int nza = 10;
    int nzb = 0;
    ComplexMatr a = rand(na, ma, nza);
    ComplexMatr b = rand(nb, mb, nzb);
    a = transp(a, na, ma);
    int tmp;
    tmp = na;
    na = ma;
    ma = tmp;
    ComplexMatr valA = value(a, na, ma);
    ComplexMatr valB = value(b, nb, mb);
    std::vector<int> rowsA = rows(a, na, ma);
    std::vector<int> rowsB = rows(b, nb, mb);
    std::vector<int> indexA = index(a, na, ma);
    std::vector<int> indexB = index(b, nb, mb);
    tmp = na;
    na = ma;
    ma = tmp;
    auto begin1 = omp_get_wtime();
    ComplexMatr CCR = umn_posled(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end1 = omp_get_wtime();
    auto posled = end1 - begin1;
    auto begin2 = omp_get_wtime();
    ComplexMatr umn = umn_parallel(valA, valB, rowsA,
    rowsB, indexA, indexB, na, ma, nb, mb);
    auto end2 = omp_get_wtime();
    auto parallel = end2 - begin2;
    for (int i = 0; i < na * mb; i++) {
        ASSERT_EQ(CCR[i], umn[i]);
    }
    std::cout << "posled time = " << posled << "s\n";
    std::cout << "parallel time = " << parallel << "s\n";
}
