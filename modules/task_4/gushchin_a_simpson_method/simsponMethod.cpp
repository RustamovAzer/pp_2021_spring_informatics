// Copyright 2021 Gushchin Artem

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "../../../3rdparty/unapproved/unapproved.h"

#include "../../../modules/task_4/gushchin_a_simpson_method/simpsonMethod.h"

namespace {
    constexpr auto numOfThreads = 4;
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

    const auto segmentsSize = segments.size();

    std::vector<double> segmentsDiffs(segmentsSize);
    std::vector<double> segmentsSteps(segmentsSize);
    std::vector<double> firstPoint(segmentsSize);
    std::vector<double> lastPoint(segmentsSize);
    double diffsProduct = 1;

    for (int i = 0; i < static_cast<int>(segmentsSize); ++i) {
        firstPoint[i] = segments[i].first;
        lastPoint[i] = segments[i].second;
        segmentsDiffs[i] = segments[i].second - segments[i].first;
        segmentsSteps[i] = segmentsDiffs[i] / stepsCount;
        diffsProduct *= segmentsDiffs[i];
    }

    diffsProduct /= stepsCount;

    using valuesPair = std::pair<double, double>;

    auto sum = [&](const int begin, const int end) {
        std::vector<double> currentPoint(segmentsSize);
        valuesPair values = { 0, 0 };

        for (auto i = begin; i < end; ++i) {
            for (size_t j = 0; j < segmentsSize; ++j)
                currentPoint[j] = firstPoint[j] + i * segmentsSteps[j];

            if (i % 2 == 0)
                values.first += func(currentPoint);
            else
                values.second += func(currentPoint);
        }

        return values;
    };

    std::vector<std::future<valuesPair>> results(numOfThreads);

    const int delta = (stepsCount - 1) / numOfThreads;
    const int remainder = (stepsCount - 1) % numOfThreads;

    int currentBegin = 1, currentEnd = 0;
    int rootBegin = 1, rootEnd = 0;

    for (int i = 0; i < numOfThreads; ++i) {
        if (i < remainder) {
            currentEnd = currentBegin + delta + 1;
        } else {
            currentEnd = currentBegin + delta;
        }

        if (i == 0) {
            rootBegin = currentBegin;
            rootEnd = currentEnd;
        } else {
            results[i] = std::async(std::launch::async,
                                    sum, currentBegin,
                                    currentEnd);
        }

        currentBegin = currentEnd;
    }

    auto values = sum(rootBegin, rootEnd);
    double evenValues = values.first, oddValues = values.second;

    for (int i = 1; i < numOfThreads; ++i) {
        results[i].wait();

        auto taskResult = results[i].get();
        evenValues += taskResult.first;
        oddValues += taskResult.second;
    }

    double coeff = diffsProduct / 3.0;
    double result = coeff * (func(firstPoint) + 2 * evenValues
        + 4 * oddValues + func(lastPoint));

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
