// Copyright 2021 Zoreev Mikhail

#include "../../../modules/task_4/zoreev_m_trapezoidal_rule/trapezoidal_rule.h"

double integrate(const std::function<double(std::vector<double>)>& f,
    const std::vector<double>& from, const std::vector<double>& to,
    uint64_t steps) {
    if (steps == 0) {
        throw std::runtime_error("Wrong steps count");
    }
    if (from.size() != to.size()) {
        throw std::runtime_error("Wrong range");
    }
    if (from.empty() || to.empty()) {
        throw std::runtime_error("Range is empty");
    }

    double block_size = 1;
    size_t blocks_count = 1;
    std::vector<double> deltas(from.size());
    std::vector<size_t>dimension_divider(from.size());

    for (size_t i = 0; i < from.size(); i++) {
        deltas[i] = (to[i] - from[i]) / static_cast<double>(steps);
        block_size *= deltas[i];
        dimension_divider[i] = blocks_count;
        blocks_count *= steps;
    }

    double result = 0.0;
    std::vector<double> block_coordinates(from.size());
    for (size_t i = 1; i < blocks_count - 1; i++) {
        for (size_t j = 0; j < from.size(); j++) {
            block_coordinates[j] = from[j] + (i / dimension_divider[j] % steps)
                * deltas[j];
        }
        result += f(block_coordinates);
    }
    result += (f(from) + f(to)) / 2;
    result *= block_size;
    return result;
}

double integrate_parallel(const std::function<double(std::vector<double>)> &f,
    const std::vector<double> &from, const std::vector<double> &to, uint64_t steps, size_t threads_count) {
    if (steps == 0) {
        throw std::runtime_error("Wrong steps count");
    }
    if (from.size() != to.size()) {
        throw std::runtime_error("Wrong range");
    }
    if (from.empty() || to.empty()) {
        throw std::runtime_error("Range is empty");
    }
    if (threads_count == 0) {
        throw std::runtime_error("Threads cant be zero count");
    }

    double block_size = 1;
    size_t blocks_count = 1;
    std::vector<double> deltas(from.size());
    std::vector<size_t>dimension_divider(from.size());

    for (size_t i = 0; i < from.size(); i++) {
        deltas[i] = (to[i] - from[i]) / static_cast<double>(steps);
        block_size *= deltas[i];
        dimension_divider[i] = blocks_count;
        blocks_count *= steps;
    }

    double result = 0.0;

    auto partial_integral = [&dimension_divider, &from, &f, &steps, &deltas]
        (size_t block_begin, size_t block_end) {
        std::vector<double> block_coordinates(from.size());
        double state_result = 0.0;
        for (size_t i = block_begin; i < block_end; i++) {
            for (size_t j = 0; j < from.size(); j++) {
                block_coordinates[j] = from[j] + (i / dimension_divider[j] % steps)
                    * deltas[j];
            }
            state_result += f(block_coordinates);
        }
        return state_result;
    };

    std::vector<std::future<double>> futures(0);
    size_t part = (blocks_count - 2) / threads_count;
    for (size_t i = 0; i < threads_count - 1; i++) {
        futures.push_back(std::async(partial_integral, 1 + i * part, 1 + (i + 1) * part));
    }
    futures.push_back(std::async(partial_integral, 1 + (threads_count - 1) * part, blocks_count - 1));

    for (size_t i = 0; i < threads_count; i++) {
        result += futures[i].get();
    }

    result += (f(from) + f(to)) / 2;
    result *= block_size;
    return result;
}
