// Copyright 2021 Kiseleva Anastasia
#include "../../../modules/task_2/kiseleva_razr_umn_openmp/umnrazr.h"

ComplexMatr rand(int n, int m, int nz) {
    ComplexMatr matr;
    std::vector<int> tmp;
    std::vector<int> num;
    for (int i = 0; i < n * m; i++) {
        tmp.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(tmp.begin(), tmp.end(), g);
    for (int i = 0; i < nz; i++) {
        num.push_back(tmp[i]);
    }
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n * m; i++) {
        int s = 0;
        for (int j = 0; j < nz; j++) {
            if (i == num[j]) {
                s = 1;
            }
        }
        if (s == 0) {
            matr.push_back(std::complex<int>(0, 0));
        } else {
            matr.push_back(std::complex<int>(gen(), gen()));
        }
    }
    return matr;
}

ComplexMatr transp(ComplexMatr a, int n, int m) {
    ComplexMatr res;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res.push_back(a[i + j * n]);
        }
    }
    return res;
}

ComplexMatr value(ComplexMatr matr, int n, int m) {
    ComplexMatr val;
    for (int i = 0; i < n * m; i++) {
        if (matr[i] != std::complex<int>(0, 0)) {
            val.push_back(matr[i]);
        }
    }
    return val;
}

std::vector<int> rows(ComplexMatr matr, int n, int m) {
    std::vector<int> row;
    std::vector<int> num;
    int p = 0;
    for (int j = 0; j < n * m; j++) {
        if (p < n) {
            num.push_back(p);
            p++;
        } else {
            p = 0;
            num.push_back(p);
            p++;
        }
    }
    for (int i = 0; i < n * m; i++) {
        if (matr[i] != std::complex<int>(0, 0)) {
            row.push_back(num[i]);
        }
    }
    return row;
}

std::vector<int> index(ComplexMatr matr, int n, int m) {
    std::vector<int> ind;
    std::vector<int> tmp;
    ind.push_back(0);
    int p = 0;
    int k = 1;
    for (int i = 0; i < n * m; i++) {
        if (i % n == 0) {
            p = 0;
        }
        if (matr[i] != std::complex<int>(0, 0)) {
            p++;
        }
        if (i != 0) {
            if (i % (n * k - 1) == 0) {
                ind.push_back(p);
                k++;
            }
        }
    }
    tmp = ind;
    int s = 1;
    for (int i = 1; i < m; i++) {
        for (int j = s + 1; j < m + 1; j++) {
            ind[j] += tmp[s];
        }
        s++;
    }
    return ind;
}

ComplexMatr umn_posled(ComplexMatr valA, ComplexMatr valB,
std::vector<int> rowsA, std::vector<int> rowsB, std::vector<int> indexA,
std::vector<int> indexB, int na, int ma, int nb, int mb) {
    ComplexMatr C;
    if (ma == nb) {
        int tmp;
        tmp = na;
        na = ma;
        ma = tmp;
        std::complex<int> tmp1(std::complex<int>(0, 0));
        for (int i = 0; i < ma; i++) {
            for (int j = 0; j < mb; j++) {
                if ((indexA[i + 1] - indexA[i] == 0) ||
                    (indexB[j + 1] - indexB[j] == 0)) {
                    C.push_back(std::complex<int>(0, 0));
                } else {
                    for (int y = indexA[i]; y < indexA[i + 1]; y++) {
                        for (int x = indexB[j]; x < indexB[j + 1]; x++) {
                            if (rowsA[y] == rowsB[x]) {
                                tmp1 += valA[y] * valB[x];
                            }
                        }
                    }
                    C.push_back(tmp1);
                    tmp1 = std::complex<int>(0, 0);
                }
            }
        }
    } else {
        throw SIZE_ERROR;
    }
    return C;
}

ComplexMatr umn_parallel(ComplexMatr valA, ComplexMatr valB,
std::vector<int> rowsA, std::vector<int> rowsB, std::vector<int> indexA,
std::vector<int> indexB, int na, int ma, int nb, int mb) {
    ComplexMatr C(na*mb);
    if (ma == nb) {
        int tmp;
        tmp = na;
        na = ma;
        ma = tmp;
        std::complex<int> tmp1(std::complex<int>(0, 0));
#pragma omp parallel
        {
#pragma omp for private(tmp1) schedule(static)
            for (int i = 0; i < ma; i++) {
                for (int j = 0; j < mb; j++) {
                    if ((indexA[i + 1] - indexA[i] == 0) ||
                        (indexB[j + 1] - indexB[j] == 0)) {
                        C[i * mb + j] = std:: complex<int>(0, 0);
                        continue;
                    } else {
                        for (int y = indexA[i]; y < indexA[i + 1]; y++) {
                            for (int x = indexB[j]; x < indexB[j + 1]; x++) {
                                if (rowsA[y] == rowsB[x]) {
                                    tmp1 += valA[y] * valB[x];
                                }
                            }
                        }
                        C[i*mb + j] = tmp1;
                        tmp1 = std::complex<int>(0, 0);
                    }
                }
            }
        }
    } else {
        throw SIZE_ERROR;
    }
    return C;
}

