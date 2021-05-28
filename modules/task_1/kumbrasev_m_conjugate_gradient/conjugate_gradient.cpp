// Copyright 2021 Kumbrasev Mark
#include "../../../modules/task_1/kumbrasev_m_conjugate_gradient/conjugate_gradient.h"

const double nzero = 1.0e-10;

std::vector<double> linVec(double a, const std::vector<double> &v1, double b, const std::vector<double> &v2) {
    int n = v1.size();
    std::vector<double> v3(n);
    for ( int i = 0; i < n; i++ ) {
        v3[i] = a * v1[i] + b * v2[i];
    }
    return v3;
}

double innerVec(const std::vector<double> &v1, const std::vector<double> &v2) {
    return inner_product( v1.begin(), v1.end(), v2.begin(), 0.0 );
}

std::vector<double> mMultv(const std::vector<std::vector<double>> &M, const std::vector<double> &V) {
    int n = M.size();
    std::vector<double> C(n);
    for ( int i = 0; i < n; i++ ) {
        C[i] = innerVec(M[i], V);
    }
    return C;
}

double vecNorm(const std::vector<double> &V) {
    return sqrt( innerVec( V, V ) );
}

std::vector<double> ConjugateGradient(const std::vector<std::vector<double>> &M, const std::vector<double> &B) {
    int n = M.size();

    std::vector<double> X(n, 0.0);

    std::vector<double> R = B;
    std::vector<double> P = R;
    int k = 0;

    while ( k < n ) {
        std::vector<double> Rold = R;
        std::vector<double> MP = mMultv(M, P);

        double alpha = innerVec(R, R) / fmax(innerVec(P, MP), nzero);
        X = linVec(1.0, X, alpha, P);
        R = linVec(1.0, R, -alpha, MP);

        if (vecNorm(R) < nzero) {
            break;
    }

    double beta = innerVec(R, R) / fmax(innerVec(Rold, Rold), nzero);
    P = linVec(1.0, R, beta, P);
    k++;
    }
    return X;
}
