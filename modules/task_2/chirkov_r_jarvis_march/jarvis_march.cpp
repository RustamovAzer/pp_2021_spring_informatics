// Copyright 2021 Chirkov Roman
#include "../../../modules/task_2/chirkov_r_jarvis_march/jarvis_march.h"

#include <omp.h>

#include <cmath>
#include <ctime>
#include <random>
#include <vector>

#define PI 3.14159265

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

bool Point::isCloserToA(Point A, Point B) {
  double sqDistAx = ((A.x - x) * (A.x - x)) + ((A.y - y) * (A.y - y));
  double sqDistBA = ((B.x - A.x) * (B.x - A.x)) + ((B.y - A.y) * (B.y - A.y));
  return (sqDistBA > sqDistAx);
}

bool operator==(Point A, Point B) { return (A.x == B.x && A.y == B.y); }

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
std::vector<Point> makePointsArrayCircle(int amount) {
  if (amount <= 0) {
    throw "Wrong amount of points!";
  }
  std::vector<Point> points;
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));

  for (int i = 0; i < amount; i++) {
    int ro, phi;
    ro = gen() % 500;
    phi = gen() % 360;

    double x, y;
    x = (ro * cos(phi * PI / 180)) + 500;
    y = (ro * sin(phi * PI / 180)) + 500;
    points.push_back(Point(x, y));
  }
  return points;
}

double rotate(Point A, Point B, Point C) {
  return (B.getX() - A.getX()) * (C.getY() - B.getY()) -
         (B.getY() - A.getY()) * (C.getX() - B.getX());
}

std::vector<int> SequentialJarvis(std::vector<Point> points) {
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

  // std::cout << "Seqindex = " << index << std::endl;

  std::vector<int> hull;
  hull.push_back(index);
  int cur = index;
  do {
    int next = cur == 0 ? 1 : 0;
    for (int i = 0; i < N; i++) {
      if (points[next] == points[i]) {
        if (next > i) next = i;
        continue;
      }
      double pos = rotate(points[cur], points[next], points[i]);
      if (pos < 0.0) {
        next = i;
      } else if (pos == 0.0) {
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

std::vector<int> ParallelJarvis(std::vector<Point> points) {
  int N = points.size();
  switch (N) {
    case 0:
      throw "wrong amount of points!";
    case 1:
      return std::vector<int>(2, 0);
  }

  int index = 0;
  Point startPoint = points[0];
#pragma omp parallel
  {
    Point loc_startPoint = startPoint;
    int loc_index = index;
#pragma omp for
    for (int i = 1; i < N; i++) {
      if (loc_startPoint == points[i]) {
        if (loc_index > i) loc_index = i;
        continue;
      }
      if ((points[i].getY() < loc_startPoint.getY()) ||
          ((loc_startPoint.getY() == points[i].getY()) &&
           (points[i].getX() < loc_startPoint.getX()))) {
        loc_startPoint = points[i];
        loc_index = i;
      }
    }
#pragma omp critical
    {
      if (loc_startPoint == startPoint) {
        if (index > loc_index) index = loc_index;
      }
      if ((loc_startPoint.getY() < startPoint.getY()) ||
          ((startPoint.getY() == loc_startPoint.getY()) &&
           (loc_startPoint.getX() < startPoint.getX()))) {
        startPoint = loc_startPoint;
        index = loc_index;
      }
    }
  }

  // std::cout << "Parindex = " << index << std::endl;

  std::vector<int> hull;
  hull.push_back(index);
  int cur = index;
  do {
    int next = (cur + 1) % N;
#pragma omp parallel
    {
      int loc_next = next;
#pragma omp for
      for (int i = 0; i < N; i++) {
        if (points[loc_next] == points[i]) {
          if (loc_next > i) loc_next = i;
          continue;
        }
        double pos = rotate(points[cur], points[loc_next], points[i]);
        if (pos < 0) {
          loc_next = i;
        } else if (pos == 0) {
          if (points[loc_next].isCloserToA(points[cur], points[i])) {
            loc_next = i;
          }
        }
      }
#pragma omp critical
      {
        if (points[next] == points[loc_next]) {
          if (next > loc_next) next = loc_next;
        } else {
          double pos = rotate(points[cur], points[next], points[loc_next]);
          if (pos < 0) {
            next = loc_next;
          } else if (pos == 0) {
            if (points[next].isCloserToA(points[cur], points[loc_next])) {
              next = loc_next;
            }
          }
        }
      }
    }
    cur = next;
    hull.push_back(next);
  } while (cur != index);

  return hull;
}
