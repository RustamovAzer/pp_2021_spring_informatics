// Copyright 2021 Gushchin Artem
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <functional>
#include <numeric>

#include "../../../modules/task_3/gushchin_a_simpson_method/simpsonMethod.h"

void pointsSum::operator()(const tbb::blocked_range<int>& r) {
    std::vector<double> currentPoint(segmentsSize);
    auto end = r.end();

    for (auto i = r.begin(); i != end; ++i) {
        for (int j = 0; j < segmentsSize; ++j)
            currentPoint[j] = firstPoint[j] + i * segmentsSteps[j];

        if (i % 2 == 0)
            evenValues += func(currentPoint);
        else
            oddValues += func(currentPoint);
    }
}

void pointsSum::join(const pointsSum& y) {
    evenValues += y.evenValues;
    oddValues += y.oddValues;
}

double parallelSimpsonsMethod(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<std::pair<double, double>>& segments,
    int stepsCount) {
    if (segments.empty())
        throw "Segments vector should contain positive number of segments";
    if (stepsCount < 1)
        throw "Steps count should be a positive value";
    if (!func)
        throw "Function is incorrect";

    if (stepsCount % 2 == 1)
        ++stepsCount;

    std::vector<double> segmentsDiffs(segments.size());
    std::vector<double> segmentsSteps(segments.size());
    std::vector<double> firstPoint(segments.size());
    std::vector<double> lastPoint(segments.size());
    double diffsProduct = 1;

    for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
        firstPoint[i] = segments[i].first;
        lastPoint[i] = segments[i].second;
        segmentsDiffs[i] = segments[i].second - segments[i].first;
        segmentsSteps[i] = segmentsDiffs[i] / stepsCount;
        diffsProduct *= segmentsDiffs[i];
    }

    diffsProduct /= stepsCount;

    pointsSum sum(static_cast<int>(segments.size()),
                  func, firstPoint, segmentsSteps);
    tbb::parallel_reduce(tbb::blocked_range<int>(1, stepsCount), sum);

    double coeff = diffsProduct / 3.0;
    double result = coeff * (func(firstPoint) + 2 * sum.evenValues
        + 4 * sum.oddValues + func(lastPoint));

    return result;
}


double simpsonsMethod(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<std::pair<double, double>>& segments,
    int stepsCount) {
    if (segments.empty())
        throw "Segments vector should contain positive number of segments";
    if (stepsCount < 1)
        throw "Steps count should be a positive value";
    if (!func)
        throw "Function is incorrect";

    if (stepsCount % 2 == 1)
        ++stepsCount;

    std::vector<double> segmentsDiffs(segments.size());
    std::vector<double> segmentsSteps(segments.size());
    std::vector<double> firstPoint(segments.size());
    std::vector<double> lastPoint(segments.size());
    double diffsProduct = 1;

    for (size_t i = 0; i < segments.size(); ++i) {
        firstPoint[i] = segments[i].first;
        lastPoint[i] = segments[i].second;
        segmentsDiffs[i] = segments[i].second - segments[i].first;
        segmentsSteps[i] = segmentsDiffs[i] / stepsCount;
        diffsProduct *= segmentsDiffs[i];
    }

    diffsProduct /= stepsCount;

    double evenValues = 0, oddValues = 0;
    std::vector<double> currentPoint = firstPoint;

    for (int i = 1; i < stepsCount; ++i) {
        for (int i = 0; i < static_cast<int>(segments.size()); ++i)
            currentPoint[i] += segmentsSteps[i];

        if (i % 2 == 0)
            evenValues += func(currentPoint);
        else
            oddValues += func(currentPoint);
    }

    double coeff = diffsProduct / 3.0;
    double result = coeff * (func(firstPoint) + 2 * evenValues
                    + 4 * oddValues + func(lastPoint));

    return result;
}
