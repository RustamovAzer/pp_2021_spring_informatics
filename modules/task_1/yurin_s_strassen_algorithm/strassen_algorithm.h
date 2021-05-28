// Copyright 2021 Yurin Stanislav
#ifndef MODULES_TASK_1_YURIN_S_STRASSEN_ALGORITHM_STRASSEN_ALGORITHM_H_
#define MODULES_TASK_1_YURIN_S_STRASSEN_ALGORITHM_STRASSEN_ALGORITHM_H_

#include <vector>

enum HorAnglePos {
    left, right
};

enum VertAnglePos {
    up, down
};

unsigned int get_random_time();
double getRandomNumber(int min, int max);

std::vector<double>::const_iterator getSubMatrConstIter(HorAnglePos, VertAnglePos);
std::vector<double> StrassenMulti(const std::vector<double>&, const std::vector<double>&);

void matrSum(std::vector<double>::const_iterator a_it,
    std::vector<double>::const_iterator b_it,
    std::vector<double>::iterator m_it,
    int m_rows_num);

void matrDiff(std::vector<double>::const_iterator a_it,
    std::vector<double>::const_iterator b_it,
    std::vector<double>::iterator m_it,
    int m_rows_num);

void matrSum(const std::vector<double>& m1,
    const std::vector<double>& m2,
    std::vector<double>::iterator c_it);

void matrDiff(const std::vector<double>& m1,
    const std::vector<double>& m2,
    std::vector<double>::iterator c_it);

void matrSum(std::vector<double>::iterator c_it,
    const std::vector<double>& m1);

void matrDiff(std::vector<double>::iterator c_it,
    const std::vector<double>& m1);

void matrMulti(std::vector<double>::const_iterator a_it,
    const std::vector<double>& p2,
    std::vector<double>::iterator m_it,
    int m_rows_num);

void matrMulti(const std::vector<double>& p2,
    std::vector<double>::const_iterator a_it,
    std::vector<double>::iterator m_it,
    int m_rows_num);

void matrMulti(const std::vector<double>& p1,
    const std::vector<double>& p2,
    std::vector<double>::iterator m_it,
    int m_rows_num);

#endif  // MODULES_TASK_1_YURIN_S_STRASSEN_ALGORITHM_STRASSEN_ALGORITHM_H_
