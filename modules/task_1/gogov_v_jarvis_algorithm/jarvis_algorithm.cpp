// Copyright 2021 Gogov Vladislav
#include <random>
#include "../../../modules/task_1/gogov_v_jarvis_algorithm/jarivs_algorithm.h"

Point::Point(int x_, int y_) {
    x = x_;
    y = y_;
}

Point::Point(const Point& temp) {
    x = temp.x;
    y = temp.y;
}

bool Point::operator<(const Point& temp) const {
    if (x < temp.x)
        return true;
    else if (x == temp.x && y < temp.y)
            return true;
    return false;
}

bool Point::operator>(const Point& temp) const {
    if (x > temp.x)
        return true;
    else if (x == temp.x && y > temp.y)
        return true;
    return false;
}

Point& Point::operator=(const Point& temp) {
    x = temp.x;
    y = temp.y;
    return *this;
}

bool Point::operator==(const Point& temp) const {
    return x == temp.x && y == temp.y;
}

bool Point::operator!=(const Point& temp) const {
    return x != temp.x || y != temp.y;
}

std::ostream& operator<<(std::ostream& out, const Point& temp) {
        out << "(" << temp.x << " , " << temp.y << ")";
        return out;
    }

int rotate(const Point& A, const Point& B, const Point& C) {
    return (C.x - A.x) * (B.y - A.y) - (C.y - A.y) * (B.x - A.x);
}

std::vector<Point> getRandomPoints(int size) {
    if (size <= 0) {
        throw("The size cannot be zero");
    }
    std::random_device device;
    std::mt19937 gen(device());
    std::vector<Point> points(size);
    for (int i = 0; i < size; i++) {
        int x = gen() % 100;
        int y = gen() % 100;
        points[i] = Point(x, y);
    }
    return points;
}

Point searchBottomLeft(const std::vector<Point>& points) {
    Point point_bottom_left = points[0];
    for (size_t i = 1; i < points.size(); i++) {
        if (points[i] < point_bottom_left)
            point_bottom_left = points[i];
    }
    return point_bottom_left;
}

bool checkPointsDistances(const Point& current, const Point& next, const Point& temp) {
    int dist_next_current = (next.x - current.x) * (next.x - current.x) + (next.y - current.y) * (next.y - current.y);
    int dist_temp_current = (temp.x - current.x) * (temp.x - current.x) + (temp.y - current.y) * (temp.y - current.y);
    if (dist_next_current < dist_temp_current)
        return true;
    return false;
}

std::vector<Point> jarvisAlgorithm(const std::vector<Point>& points) {
    size_t count_points = points.size();
    if (count_points == 0) {
        throw("It is impossible to construct a convex hull");
    }
    if (count_points < 2) {
        return points;
    }
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
