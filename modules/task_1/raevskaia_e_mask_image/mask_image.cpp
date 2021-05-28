// Copyright 2021 Ekaterina Raevskaya
#include "../../../modules/task_1/raevskaia_e_mask_image/mask_image.h"
#include <stack>

const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};

std::vector <char> used;

bool valid(int x, int y, int n, int m) {
    return x >= 0 && x < n && y >= 0 && y < m;
}

void dfs(std::vector<int>* image, int x, int y, int n, int m,
         int value, std::vector<int>* res_image) {
    std::stack<std::pair<int, int>> s;
    s.push(std::make_pair(x, y));
    used[x*m + y] = true;
    (*res_image)[x*m + y] = value;
    while (!s.empty()) {
        std::pair<int, int> v = s.top();
        s.pop();
        x = v.first;
        y = v.second;
        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (valid(nx, ny, n, m) && !used[nx*m + ny] &&
                (*image)[nx*m + ny]) {
                s.push(std::make_pair(nx, ny));
                used[nx*m + ny] = true;
                (*res_image)[nx*m + ny] = value;
            }
        }
    }
}

std::vector<int> mask(std::vector<int>* image, int n, int m) {
    used.resize(n*m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            used[i*m + j] = false;
        }
    }

    int value = 1;

    std::vector<int> result(n*m, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (!used[i*m + j] && (*image)[i*m + j]) {
                dfs(image, i, j, n, m, value, &result);
                ++value;
            }
        }
    }

    return result;
}
