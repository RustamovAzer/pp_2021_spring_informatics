// Copyright 2021 Gurylev Nikita
#include <random>
#include <vector>
#include <ctime>
#include <utility>

#include "../../../modules/task_1/gurylev_n_convex_hull_jarvis/convex_hull_jarvis.h"

std::vector<std::pair<int, int>> getRandomPoint(int size) {
    if (size <= 0) {
        throw("Invalid size!");
    }
    std::vector<std::pair<int, int>> points(size);
    std::default_random_engine random;
    random.seed(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < size; i++) {
        int x = random() % 100;
        int y = random() % 100;
        points[i] = std::make_pair(x, y);
    }
    return points;
}

std::pair<int, int> SLeftMPoint(const std::vector<std::pair<int,
    int>>& points) {
    std::pair<int, int> point_left = points[0];
    for (size_t i = 1; i < points.size(); i++) {
        if (points[i] < point_left)
            point_left = points[i];
    }
    return point_left;
}

bool Dist(const std::pair<int, int> curr,
    const std::pair<int, int> next, const std::pair<int,
    int> tmp) {
    int dist_next_curr = (next.first - curr.first) * (next.first - curr.first) +
        (next.second - curr.second) * (next.second - curr.second);
    int dist_tmp_curr = (tmp.first - curr.first) * (tmp.first - curr.first) +
        (tmp.second - curr.second) * (tmp.second - curr.second);
    if (dist_next_curr < dist_tmp_curr)
        return true;
    return false;
}

double Around(const std::pair<int, int>& A,
    const std::pair<int, int>& B, const std::pair<int, int>& C) {
    return (C.first - A.first) * (B.second - A.second) -
        (C.second - A.second) * (B.first - A.first);
}

std::vector < std::pair<int, int>> JarvisAlg(const std::vector<std::pair<int,
    int>>& points) {
    size_t count_points = points.size();
    if (count_points == 0) {
        throw("Error");
    }
    if (count_points < 2) {
        return points;
    }
    std::pair<int, int> base = SLeftMPoint(points);
    std::vector < std::pair<int, int>> convex_hull;
    convex_hull.push_back(base);

    std::pair<int, int> curr = base;
    std::pair<int, int> next;
    do {
        next = curr == points[0] ? points[1] : points[0];

        for (size_t i = 0; i < count_points; i++) {
            double tmp = Around(curr, next, points[i]);
            if (tmp > 0) {
                next = points[i];
            } else if (tmp == 0) {
                if (Dist(curr, next, points[i])) {
                    next = points[i];
                }
            }
        }
        curr = next;
        convex_hull.push_back(next);
    } while (curr != base);
    convex_hull.pop_back();
    return convex_hull;
}
