// Copyright 2021 Gogov Vladislav

#include <functional>
#include <random>

#include "../../../modules/task_4/gogov_v_jarvis_algorithm/jarivs_algorithm.h"
#include "../../../3rdparty/unapproved/unapproved.h"

int rotate(const Point& A, const Point& B, const Point& C) {
    int64_t temp = (C.first - A.first) * (B.second - A.second) -
                    (C.second - A.second) * (B.first - A.first);
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
        int64_t x = static_cast<int64_t>(gen() % 1000);
        int64_t y = static_cast<int64_t>(gen() % 1000);
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
    int64_t dist_next_current =
        (next.first - current.first) * (next.first - current.first) +
        (next.second - current.second) * (next.second - current.second);
    int64_t dist_temp_current =
        (temp.first - current.first) * (temp.first - current.first) +
        (temp.second - current.second) * (temp.second - current.second);
    if (dist_next_current < dist_temp_current)
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

std::mutex mut;

std::vector<Point> jarvisAlgorithmStd(const std::vector<Point>& points) {
    int count_points = static_cast<int>(points.size());
    if (count_points == 0)
        throw("It is impossible to construct a convex hull");
    if (count_points < 2)
        return points;

    const int num_threads = std::thread::hardware_concurrency();
    int delta = count_points / num_threads;
    std::vector<int> limits(num_threads + 1);
    for (int i = 0; i < num_threads; i++) {
        limits[i] = i * delta;
    }
    limits[num_threads] = count_points;

    Point base = points[0];
    auto base_point = [&points, &limits, &base] (int id_thread) -> const Point& {
        std::reference_wrapper<const Point> local_base(base);
        for (int i = limits[id_thread]; i < limits[id_thread + 1]; i++) {
            if (points[i] < local_base.get())
                local_base = points[i];
        }
        return local_base;
    };

    std::vector<std::future<const Point&>> result_base_points;
    for (int i = 0; i < num_threads; i++) {
        result_base_points.push_back(std::async(base_point, i));
    }

    for (auto& result_base_point : result_base_points) {
        auto local_base = result_base_point.get();
        if (local_base < base)
            base = local_base;
    }

    std::vector<Point> convex_hull;
    convex_hull.push_back(base);
    Point current = base;
    Point next;

    auto next_point = [&points, &limits, &next, &current] (int id_thread) -> const Point& {
        std::reference_wrapper<const Point> local_next(next);
            for (int i = limits[id_thread]; i < limits[id_thread + 1]; i++) {
                int temp = rotate(current, local_next.get(), points[i]);
                if (temp > 0) {
                    local_next = points[i];
                } else if (temp == 0) {
                    if (checkPointsDistances(current, local_next.get(), points[i])) {
                        local_next = points[i];
                    }
                }
            }
        return local_next;
    };

    std::vector<std::future<const Point&>> result_next_points(num_threads);
    do {
        next = current == points[0] ? points[1] : points[0];

        for (int i = 0; i < num_threads; i++) {
            result_next_points[i] = std::async(next_point, i);
        }

        for (auto& result_next_point : result_next_points) {
            auto local_next = result_next_point.get();
            int temp = rotate(current, next, local_next);
            if (temp > 0) {
                next = local_next;
            } else if (temp == 0) {
                if (checkPointsDistances(current, next, local_next)) {
                    next = local_next;
                }
            }
        }

        current = next;
        convex_hull.push_back(next);
    } while (current != base);
    convex_hull.pop_back();
    return convex_hull;
}
