// Copyright 2021 Ekaterina Raevskaya
#include <omp.h>
#include "../../../modules/task_2/raevskaya_e_mask_image_omp/mask_image.h"

const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};

std::vector <int> parent;
std::vector <int> ranks;
std::vector <char> used;

void make_set(int v) {
    parent[v] = v;
    ranks[v] = 0;
}

int find_set(int v) {
    if (v == parent[v]) {
        return v;
    }
    return parent[v] = find_set(parent[v]);
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (ranks[a] < ranks[b]) {
            std::swap(a, b);
        }
        parent[b] = a;
        if (ranks[a] == ranks[b]) {
            ++ranks[a];
        }
    }
}

bool valid(int x, int y, int n, int m) {
    return x >= 0 && x < n && y >= 0 && y < m;
}

std::vector<int> mask_dsu(std::vector<int>* image, int n, int m) {
    parent.resize(n*m);
    ranks.resize(n*m);
    for (int i = 0; i < n*m; i++) {
        make_set(i);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if ((*image)[i*m + j]) {
                for (int k = 0; k < 4; k++) {
                    int nx = i + dx[k];
                    int ny = j + dy[k];
                    if (valid(nx, ny, n, m) && (*image)[nx*m + ny]) {
                        union_sets(i*m + j, nx*m + ny);
                    }
                }
            }
        }
    }

    std::vector <int> result(n*m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if ((*image)[i*m + j]) {
                result[i*m + j] = find_set(i*m + j) + 1;
            }
        }
    }

    return result;
}

std::pair<std::vector<int>, int> mask_dsu_omp(std::vector<int>* image, int n, int m) {
    parent.resize(n*m);
    ranks.resize(n*m);
    for (int i = 0; i < n*m; i++) {
        make_set(i);
    }

    int ans = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if ((*image)[i*m + j]) {
                for (int k = 0; k < 4; k++) {
                    int nx = i + dx[k];
                    int ny = j + dy[k];
                    if (valid(nx, ny, n, m) && (*image)[nx*m + ny]) {
                        union_sets(i*m + j, nx*m + ny);
                    }
                }
            }
        }
    }

    std::vector <int> result(n*m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if ((*image)[i*m + j]) {
                result[i*m + j] = find_set(i*m + j) + 1;
            }
            if (result[i*m + j] == i*m + j + 1) {
                ans++;
            }
        }
    }

    return {result, ans};
}
