// Copyright 2021 Gushchin Artem
#ifndef MODULES_TASK_3_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_
#define MODULES_TASK_3_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_

#include <tbb/blocked_range.h>

#include <utility>
#include <vector>

class pointsSum {
    int segmentsSize;
    const std::function<double(const std::vector<double>&)>& func;
    const std::vector<double>& firstPoint;
    const std::vector<double>& segmentsSteps;

 public:
    double evenValues = 0;
    double oddValues = 0;

    void operator()(const tbb::blocked_range<int>& r);

    pointsSum(const pointsSum& x, tbb::split) :
        segmentsSize(x.segmentsSize), func(x.func),
        firstPoint(x.firstPoint), segmentsSteps(x.segmentsSteps) {}

    void join(const pointsSum& y);

    pointsSum(int segmentsSize,
        const std::function<double(const std::vector<double>&)>& func,
        const std::vector<double>& firstPoint,
        const std::vector<double>& segmentsSteps) :
        segmentsSize(segmentsSize), func(func),
        firstPoint(firstPoint), segmentsSteps(segmentsSteps)
    {}
};


double parallelSimpsonsMethod(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<std::pair<double, double>>& segments,
    int stepsCount);


double simpsonsMethod(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<std::pair<double, double>>& segments,
    int stepsCount);

#endif  // MODULES_TASK_3_GUSHCHIN_A_SIMPSON_METHOD_SIMPSONMETHOD_H_
