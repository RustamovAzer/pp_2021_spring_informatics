// Copyright 2021 Chirkov Roman
#include "../../../modules/task_3/chirkov_r_jarvis_march/jarvis_march.h"

#include <cmath>
#include <ctime>
#include <random>
#include <vector>

#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"

// #include "tbb/tick_count.h"  // std::min crash

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

  // tbb::task_scheduler_init tsi(1);
  // tbb::tick_count t0 = tbb::tick_count::now();

  int index = tbb::parallel_reduce(
      tbb::blocked_range<int>(1, N, 6250), 0,
      [&](tbb::blocked_range<int>& r, int loc_index) -> int {
        Point loc_startPoint = points[loc_index];
        for (auto i = r.begin(); i != r.end(); i++) {
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
        return loc_index;
      },
      [&](int indexA, int indexB) -> int {
        if (points[indexA] == points[indexB]) {
          return indexA < indexB ? indexA : indexB;
        }
        if ((points[indexA].getY() < points[indexB].getY()) ||
            ((points[indexA].getY() == points[indexB].getY()) &&
             (points[indexA].getX() < points[indexB].getX()))) {
          return indexA;
        }
        return indexB;
      });

  // tbb::tick_count t1 = tbb::tick_count::now();
  // printf("work took %g seconds\n", (t1 - t0).seconds());
  // tsi.terminate();

  std::vector<int> hull;
  hull.push_back(index);

  int cur = index;
  do {
    int next = (cur + 1) % N;
    // tsi.initialize(1);
    next = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N, 6250), next,
        [&](tbb::blocked_range<int>& r, int loc_next) -> int {
          for (int i = r.begin(); i < r.end(); i++) {
            if (points[loc_next] == points[i]) {
              if (loc_next > i) loc_next = i;
              continue;
            }
            double pos = rotate(points[cur], points[loc_next], points[i]);
            if (pos < 0.0) {
              loc_next = i;
            } else if (pos == 0.0) {
              if (points[loc_next].isCloserToA(points[cur], points[i])) {
                loc_next = i;
              }
            }
          }
          return loc_next;
        },
        [&](int indexA, int indexB) -> int {
          if (points[indexA] == points[indexB]) {
            return indexA < indexB ? indexA : indexB;
          }
          double pos = rotate(points[cur], points[indexA], points[indexB]);
          if (pos < 0.0) {
            return indexB;
          }
          if (pos == 0.0) {
            if (points[indexA].isCloserToA(points[cur], points[indexB])) {
              return indexB;
            }
          }

          return indexA;
        });
    // tsi.terminate();
    cur = next;
    hull.push_back(next);
  } while (cur != index);
  return hull;
}
