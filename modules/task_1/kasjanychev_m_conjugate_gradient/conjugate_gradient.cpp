// Copyright 2021 Kasjanychev Mikhail
#include "../../../modules/task_1/kasjanychev_m_conjugate_gradient/conjugate_gradient.h"

std::vector<double> multiMtxVec(std::vector<std::vector<double> > mtx, std::vector<double> x) {
    assert(mtx.size() > 0 && x.size() > 0);
    assert(mtx.size() % x.size() == 0);
    std::vector<double> res;
    for (size_t i = 0; i < mtx.size(); ++i) {
        double tmp = 0;
        for (size_t j = 0; j < x.size(); ++j) {
            tmp += mtx[i][j] * x[j];
        }
        res.push_back(tmp);
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
    for (size_t i = 0; i < a.size(); i++) {
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

std::vector<double> calculateRes(std::vector<std::vector<double> > mtx, const std::vector<double>& b) {
    int n = mtx.size();
    assert(n > 0);
    int m = mtx[0].size();
    std::vector<double> x0;
    std::vector<double> r0;
    std::vector<double> z0 = minusVec(b);
    for (int i = 0; i < m; i++) {
        x0.push_back(0);
    }
    r0 = subVec(b, multiMtxVec(mtx, x0));
    z0 = r0;
    int i = 0;
    do {
        i++;
        double alpha = (scalarProduct(r0, r0) / (scalarProduct(multiMtxVec(mtx, z0), z0)));
        std::vector<double> x = sumVec(x0, multiVec(alpha, z0));
        std::vector<double> r = subVec(r0, multiVec(alpha, multiMtxVec(mtx, z0)));
        double beta = scalarProduct(r, r) / scalarProduct(r0, r0);
        std::vector<double> z = sumVec(r, multiVec(beta, z0));
        x0 = x;
        r0 = r;
        z0 = z;
    } while (i < n);
    return x0;
}
