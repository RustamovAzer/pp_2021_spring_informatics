// Copyright 2021 Gogov Vladislav
#include <random>
#include <ostream>
#include "../../../modules/task_3/gogov_v_jarvis_algorithm/jarivs_algorithm.h"

int rotate(const Point& A, const Point& B, const Point& C) {
    int64_t temp = (C.first - A.first) * (B.second - A.second) - (C.second - A.second) * (B.first - A.first);
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}

std::vector<Point> getRandomPoints(size_t size) {
    if (size <= 0)
        throw("The size cannot be zero");
    std::random_device device;
    std::mt19937 gen(device());
    std::vector<Point> points(size);
    for (size_t i = 0; i < size; i++) {
        int64_t x = static_cast<int64_t>(gen() % 1000ull);
        int64_t y = static_cast<int64_t>(gen() % 1000ull);
        points[i] = Point(x, y);
    }
    return points;
}

Point searchBottomLeft(const std::vector<Point>& points) {
    Point point_bottom_left = points[0];
    for (size_t i = 1; i < points.size(); i++)
        if (points[i] < point_bottom_left)
            point_bottom_left = points[i];
    return point_bottom_left;
}

bool checkPointsDistances(const Point& current, const Point& next, const Point& temp) {
    int64_t dist_nefirstt_current = (next.first - current.first) * (next.first - current.first) +
                                   (next.second - current.second) * (next.second - current.second);
    int64_t dist_temp_current = (temp.first - current.first) * (temp.first - current.first) +
                               (temp.second - current.second) * (temp.second - current.second);
    if (dist_nefirstt_current < dist_temp_current)
        return true;
    return false;
}

std::vector<Point> jarvisAlgorithmSeq(const std::vector<Point>& points) {
    size_t count_points = points.size();
    if (count_points == 0)
        throw("It is impossible to construct a convex hull");
    if (count_points < 2)
        return points;
    Point base = searchBottomLeft(points);
    std::vector<Point> convex_hull;
    convex_hull.push_back(base);
    Point current = base;
    Point next;
    do {
        next = current == points[0] ? points[1] : points[0];

        for (size_t i = 0; i < count_points; i++) {
            int temp = rotate(current, next, points[i]);
            if (temp > 0) {
                next = points[i];
            } else if (temp == 0) {
                if (checkPointsDistances(current, next, points[i])) {
                    next = points[i];
                }
            }
        }
        current = next;
        convex_hull.push_back(next);
    } while (current != base);
    convex_hull.pop_back();
    return convex_hull;
}

std::vector<Point> jarvisAlgorithmTbb(const std::vector<Point>& points) {
    int count_points = static_cast<int>(points.size());
    if (count_points == 0)
        throw("It is impossible to construct a convex hull");
    if (count_points < 2)
        return points;

    Point base = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(1, count_points),
        points[0],
        [&points] (tbb::blocked_range<size_t>& r, Point local_base) -> Point {
            auto begin = r.begin(), end = r.end();
            for (auto i = begin; i != end; i++) {
                if (points[i] < local_base)
                    local_base = points[i];
            }
            return local_base;
        },
        [] (const Point& a, const Point& b) -> Point {
            return a < b ? a : b;
        });

    std::vector<Point> convex_hull;
    convex_hull.push_back(base);
    Point current = base;
    Point next;
    do {
        next = current == points[0] ? points[1] : points[0];

        current = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, count_points),
            next,
            [&current, &points] (tbb::blocked_range<size_t>& r, Point local_next) -> Point {
                auto begin = r.begin(), end = r.end();
                for (auto i = begin; i != end; i++) {
                    int temp = rotate(current, local_next, points[i]);
                    if (temp > 0) {
                        local_next = points[i];
                    } else if (temp == 0) {
                        if (checkPointsDistances(current, local_next, points[i]))
                            local_next = points[i];
                    }
                }
                return local_next;
            },
            [&current] (const Point& next, const Point& local_next) -> Point {
                int temp = rotate(current, next, local_next);
                if (temp > 0) {
                    return local_next;
                } else if (temp == 0) {
                    if (checkPointsDistances(current, next, local_next))
                        return local_next;
                }
                return next;
            });

        next = current;
        convex_hull.push_back(next);
    } while (current != base);
    convex_hull.pop_back();
    return convex_hull;
}
