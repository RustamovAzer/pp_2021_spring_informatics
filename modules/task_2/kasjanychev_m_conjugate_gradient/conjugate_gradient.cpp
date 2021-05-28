// Copyright 2021 Kasjanychev Mikhail
#include "../../../modules/task_2/kasjanychev_m_conjugate_gradient/conjugate_gradient.h"

std::vector<std::vector<double> > getRandomMatrix(int n) {
    std::vector<std::vector<double> > res(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i][j] = std::rand() % 100 - 100;
        }
    }
    res = multiMtx(sumMtx(res, transposeMtx(res)), 0.5);
    return res;
}

std::vector<double> getRandomVector(int n) {
    std::vector<double> res(n);
    for (int i = 0; i < n; i++) {
        res[i] = std::rand() % 100 - 100;
    }
    return res;
}

std::vector<double> multiMtxVec(std::vector<std::vector<double> > mtx, std::vector<double> x) {
    assert(mtx.size() > 0 && x.size() > 0);
    assert(mtx.size() % x.size() == 0);
    std::vector<double> res(x);
    int n = mtx.size();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double tmp = 0;
        for (size_t j = 0; j < x.size(); ++j) {
            tmp += mtx[i][j] * x[j];
        }
        res[i] = tmp;
    }
    return res;
}

std::vector<double> subVec(std::vector<double> a, std::vector<double> b) {
    assert(a.size() == b.size());
    std::vector<double> res;
    for (size_t i = 0; i < a.size(); i++) {
        double tmp;
        tmp = a[i] - b[i];
        res.push_back(tmp);
    }
    return res;
}

std::vector<double> sumVec(std::vector<double> a, std::vector<double> b) {
    assert(a.size() == b.size());
    std::vector<double> res;
    for (size_t i = 0; i < a.size(); i++) {
        double tmp;
        tmp = a[i] + b[i];
        res.push_back(tmp);
    }
    return res;
}

double scalarProduct(std::vector<double> a, std::vector<double> b) {
    assert(a.size() == b.size());
    double res = 0;
    int n = a.size();
    for (int i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    return res;
}

std::vector<double> multiVec(double a, std::vector<double> x) {
    std::vector<double> res;
    for (size_t i = 0; i < x.size(); i++) {
        res.push_back(x[i] * a);
    }
    return res;
}

std::vector<double> minusVec(std::vector<double> x) {
    std::vector<double> res;
    for (size_t i = 0; i < x.size(); i++) {
        res.push_back(-x[i]);
    }
    return res;
}

std::vector<std::vector<double> > transposeMtx(std::vector<std::vector<double> > mtx) {
    std::vector<std::vector<double> > res = mtx;
    int n = mtx.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            res[i][j] = mtx[j][i];
        }
    return res;
}

std::vector<std::vector<double> > sumMtx(std::vector<std::vector<double> > mtx1,
    std::vector<std::vector<double> > mtx2) {
    std::vector<std::vector<double> > res = mtx1;
    int n = mtx1.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            res[i][j] = mtx1[i][j] + mtx2[i][j];
        }
    return res;
}

std::vector<std::vector<double> > multiMtx(std::vector<std::vector<double> > mtx, double a) {
    std::vector<std::vector<double> > res = mtx;
    int n = mtx.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            res[i][j] *= a;
        }
    return res;
}

std::vector<std::vector<double> > searchReverseMatrix(std::vector<std::vector<double> > matrix) {
    int size = matrix.size();
    std::vector<std::vector<double> > E(size, std::vector<double>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                E[i][j] = 1.0;
            } else {
                E[i][j] = 0.0;
            }
        }
    }
    for (int k = 0; k < size; k++) {
        if (abs(matrix[k][k]) < 1e-8) {
            for (int i = k + 1; i < size; i++) {
                if (abs(matrix[i][k]) > 1e-8) {
                    swap(matrix[k], matrix[i]);
                    swap(E[k], E[i]);
                    break;
                }
            }
        }

        double div = matrix[k][k];
#pragma omp parallel for
            for (int j = 0; j < size; j++) {
                matrix[k][j] /= div;
                E[k][j] /= div;
            }
#pragma omp parallel for
            for (int i = k + 1; i < size; i++) {
                double multi = matrix[i][k];


                for (int j = 0; j < size; j++) {
                    matrix[i][j] -= multi * matrix[k][j];
                    E[i][j] -= multi * E[k][j];
                }
            }
        }
    for (int k = size - 1; k > 0; k--) {
#pragma omp parallel for
            for (int i = k - 1; i > -1; i--) {
                double multi = matrix[i][k];

                for (int j = 0; j < size; j++) {
                    matrix[i][j] -= multi * matrix[k][j];
                    E[i][j] -= multi * E[k][j];
                }
            }
    }
    matrix = E;
    return matrix;
}

std::vector<double> calculateStandardRes(const std::vector<std::vector<double> >& _mtx,
    std::vector<double> _b, int proc) {
    omp_set_num_threads(proc);
    std::vector<std::vector<double> > matrix = _mtx;
    std::vector<double> B = _b;
    int equations_amount = _b.size();
    matrix = searchReverseMatrix(matrix);
    std::vector<double> X(equations_amount);
#pragma omp parallel for
        for (int i = 0; i < equations_amount; i++) {
            X[i] = 0;
            for (int j = 0; j < equations_amount; j++)
                X[i] += matrix[i][j] * B[j];
        }
    return X;
}

std::vector<double> calculateRes(std::vector<std::vector<double> > mtx, const std::vector<double>& b, int proc) {
    int m = mtx[0].size();
    assert(m > 0);
    std::vector<double> x0;
    std::vector<double> r0;
    std::vector<double> z0 = minusVec(b);
    double check = 0.0;
    double check1 = 0.01;
    for (int i = 0; i < m; i++) {
        x0.push_back(0);
    }
    r0 = subVec(b, multiMtxVec(mtx, x0));
    z0 = r0;
    int i = 0;
    omp_set_num_threads(proc);
    do {
        i++;
        double alpha = (scalarProduct(r0, r0) / (scalarProduct(multiMtxVec(mtx, z0), z0)));
        std::vector<double> x = sumVec(x0, multiVec(alpha, z0));
        std::vector<double> r = subVec(r0, multiVec(alpha, multiMtxVec(mtx, z0)));
        double beta = scalarProduct(r, r) / scalarProduct(r0, r0);
        std::vector<double> z = sumVec(r, multiVec(beta, z0));
        check = sqrt(scalarProduct(r, r));
        x0 = x;
        r0 = r;
        z0 = z;
    } while (check > check1);
    return x0;
}
