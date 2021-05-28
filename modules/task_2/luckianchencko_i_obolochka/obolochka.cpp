// Copyright 2021 Luckianchencko Ivan
#include <cmath>
#include <ctime>
#include <vector>
#include <stack>
#include<iostream>
#include<random>
#include<algorithm>
#include "../../../modules/task_2/luckianchencko_i_obolochka/obolochka.h"

mat get_rand_mat(int size) {
    if (size <= 0)
        throw std::invalid_argument("Error size of matrix");
    std::mt19937 gen;
    gen.seed(static_cast<int>(time(0)));
    mat res(size, std::vector<int>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            res[i][j] = gen() % 2;
        }
    }
    return res;
}
double distance(point a, point b) {
    double res = sqrt(pow(b.x-a.x, 2) + pow(b.y-a.y, 2));
    return res;
}
double cosAngle(point a, point b, point c) {
    double ab = distance(a, b);
    double bc = distance(b, c);
    if ( (ab > 0) && (bc > 0) ) {
        double cos = ((b.x-a.x) * (b.x -c.x) + (b.y - a.y) * (b.y - c.y))
        /(ab*bc);
        return cos;
    } else {
        return 10.0;
    }
}
std::vector<point> get_points(mat img, int size) {
    std::vector<point> res;
    point a;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (img[i][j] == 1) {
               a.x = i;
               a.y = j;
               res.push_back(a);
            }
        }
    }
    return res;
}
std::vector<point> Paral_Jarvis_alg(const mat &img, int size) {
    std::vector<point> a = get_points(img, size);
    std::vector<point> res;
    int count = a.size();
    int base = 0;
#pragma omp parallel shared(a)
    {
    int local_base = base;
#pragma omp for
    for (int i = 0; i < count; i++) {
        if (a[i].x > a[local_base].x)
            local_base = i;
        else
            if ( (a[i].x == a[local_base].x) && (a[i].y < a[local_base].y) )
               local_base = i;
    }
#pragma omp critical
    {
        if (a[local_base].x > a[base].x)
        base = local_base;
        else
            if ( (a[local_base].x == a[base].x) &&
                (a[local_base].y < a[base].y) )
               base = local_base;
    }
}
    res.push_back(a[base]);
    point first = a[base];
    point prev = first;
    prev.y -= 1;
    point cur = first;
    int num = 0;
    while (1) {
        double min = 2;
        double dist = 0;
#pragma omp parallel shared(a, num, min , dist)
    {
        count = a.size();
        double loc_dist = 0;
        double loc_min = 2;
        int loc_num = num;
#pragma omp for
        for ( int i = 0; i < count; i++ ) {
            if (std::abs(cosAngle(prev, cur, a[i]) - loc_min) < 1e-5) {
                double curdist = distance(cur, a[i]);
                if (curdist > loc_dist) {
                    loc_min = cosAngle(prev, cur, a[i]);
                    loc_dist = distance(cur, a[i]);
                    loc_num = i;
                }
            } else if ( cosAngle(prev, cur, a[i]) < loc_min ) {
                loc_min = cosAngle(prev, cur, a[i]);
                loc_dist = distance(cur, a[i]);
                loc_num = i;
                }
        }
#pragma omp critical
    {
        if ( std::abs(loc_min - min) < 1e-5 ) {
            if ( loc_dist > dist ) {
                num = loc_num;
                min = loc_min;
                dist = loc_dist;
            }
        } else if ( loc_min < min ) {
            min = loc_min;
            num = loc_num;
            dist = loc_dist;
        }
    }
    }
    prev = cur;
    cur = a[num];
    res.push_back(a[num]);
    a.erase(a.begin()+num);
    if ( ( cur.x == first.x ) && ( cur.y == first.y ) )
        break;
    }
  return res;
}

std::vector<point> Jarvis_alg(const mat &img, int size) {
    std::vector<point> a = get_points(img, size);
    std::vector<point> res;
    int count = a.size();
    int base = 0;
    for (int i = 0; i < count; ++i) {
        if (a[i].x > a[base].x)
            base = i;
        else if ((a[i].x == a[base].x) && (a[i].y < a[base].y))
            base = i;
    }
    res.push_back(a[base]);
    point first = a[base];
    point prev = first;
    prev.y -= 1;
    point cur = first;
    int num = -1;
    while (1) {
        double min = 2;
        double dist = 0;
        count = a.size();
        for (int i = 0; i < count; i++) {
            if (std::abs(cosAngle(prev, cur, a[i]) - min) < 1e-5) {
                double curdist = distance(cur, a[i]);
                if (curdist > dist) {
                    min = cosAngle(prev, cur, a[i]);
                    dist = curdist;
                    num = i;
                }
            } else if (cosAngle(prev, cur, a[i]) < min) {
                min = cosAngle(prev, cur, a[i]);
                dist = distance(cur, a[i]);
                num = i;
            }
        }
        res.push_back(a[num]);
        prev = cur;
        cur = a[num];
        a.erase(a.begin()+num);
        if ((cur.x == first.x) && (cur.y == first.y)) {
            break;
        }
    }
    return res;
}
