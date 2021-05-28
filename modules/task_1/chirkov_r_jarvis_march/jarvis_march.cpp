// Copyright 2021 Chirkov Roman
#include <random>
#include <vector>
#include <ctime>
#include "../../../modules/task_1/chirkov_r_jarvis_march/jarvis_march.h"

Point::Point() {
  x = 0;
  y = 0;
}
Point::Point(double _x, double _y) {
  x = _x;
  y = _y;
}
double Point::getX() { return x; }
double Point::getY() { return y; }
void Point::setX(double _x) { x = _x; }
void Point::setY(double _y) { y = _y; }
bool operator==(Point p1, Point p2) { return (p1.x == p2.x && p1.y == p2.y); }

std::vector<Point> makePointsArray(int amount) {
  if (amount <= 0) {
    throw "Wrong amount of points!";
  }
  std::vector<Point> points;
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));

  for (int i = 0; i < amount; i++) {
    points.push_back(Point(gen() % 1000, gen() % 1000));
  }
  return points;
}

double rotate(Point A, Point B, Point C) {
  return (B.getX() - A.getX()) * (C.getY() - B.getY()) -
         (B.getY() - A.getY()) * (C.getX() - B.getX());
}

bool Point::isCloserToA(Point A, Point B) {
  return (A.getX() <= x && x <= B.getX() && A.getY() <= y && y <= B.getY());
}

std::vector<int> Jarvis(std::vector<Point> points) {
  int N = points.size();
  switch (N) {
    case 0:
      throw "wrong amount of points";
    case 1:
      return std::vector<int>(2, 0);
  }

  int index = 0;
  Point startPoint = points[0];
  for (int i = 1; i < N; i++) {
    Point next = points[i];
    if (startPoint == next) {
      continue;
    }
    if ((next.getY() < startPoint.getY()) ||
        ((startPoint.getY() == next.getY()) &&
         (next.getX() < startPoint.getX()))) {
      startPoint = next;
      index = i;
    }
  }

  std::vector<int> hull;
  hull.push_back(index);
  int cur = index;
  do {
    int next = cur == 0 ? 1 : 0;
    for (int i = 0; i < N; i++) {
      double pos = rotate(points[cur], points[next], points[i]);
      if (pos < 0) {
        next = i;
      } else if (pos == 0) {
        if (points[next].isCloserToA(points[cur], points[i])) {
          next = i;
        }
      }
    }
    cur = next;
    hull.push_back(next);
  } while (cur != index);
  return hull;
}
