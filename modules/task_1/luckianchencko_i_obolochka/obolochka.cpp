// Copyright 2021 Luckianchencko Ivan
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stack>
#include "../../../modules/task_1/luckianchencko_i_obolochka/obolochka.h"

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
std::vector<point> get_points(mat img) {
    std::vector<point> res;
    point a;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (img[j][i] == 1) {
               a.x = j;
               a.y = i;
               res.push_back(a);
            }
        }
    }
    return res;
}
std::vector<point> Jarvis_alg(const mat &img) {
    std::vector<point> a = get_points(img);
    std::vector<point> res;
    int count = a.size();
    int base = 0;
    for (int i = 0; i < count; ++i) {
        if (a[i].y < a[0].y)
            base = i;
        else
            if ( (a[i].y == a[base].y) && (a[i].x < a[base].x) )
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
        for (int i = 0; i < count; i++) {
            if (cosAngle(prev, cur, a[i]) < min) {
                min = cosAngle(prev, cur, a[i]);
                dist = distance(cur, a[i]);
                num = i;
                } else if (std::abs(cosAngle(prev, cur, a[i]) - min) < 1e5) {
                double curdist = distance(cur, a[i]);
                if (curdist > dist) {
                    min = cosAngle(prev, cur, a[i]);
                    dist = distance(cur, a[i]);
                    num = i;
                }
            }
        }
        res.push_back(a[num]);
        prev = cur;
        cur = a[num];
        if ( (cur.x == first.x) && (cur.y == first.y) )
            break;
  }
  res.pop_back();
  return res;
}



