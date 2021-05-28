// Copyright 2021 Yurin Stanislav
#include <vector>
#include <random>
#include <iostream>
#include "../../../modules/task_1/yurin_s_strassen_algorithm/strassen_algorithm.h"

unsigned int time_increase = static_cast<unsigned int>(time(0));

unsigned int get_random_time() {
    return time_increase++;
}

double getRandomNumber(int min, int max) {
    std::mt19937 gen;
    gen.seed(get_random_time());
    int a = (gen()) % ((max + 1) - min);
    int b = a + min;

    return b;
}

std::vector<double>::const_iterator getLeftUpSubMatrConstIter(const std::vector<double>& matr) {
    return matr.begin();
}

std::vector<double>::const_iterator getLeftDownSubMatrConstIter(const std::vector<double>& matr) {
    return matr.begin() + (matr.size() / 2);
}

std::vector<double>::const_iterator getRightUpSubMatrConstIter(const std::vector<double>& matr) {
    int row_num = sqrt(matr.size());

    return matr.begin() + (row_num / 2);
}

std::vector<double>::const_iterator getRightDownSubMatrConstIter(const std::vector<double>& matr) {
    int row_num = sqrt(matr.size());

    return matr.begin() + (matr.size() / 2) + (row_num / 2);
}


std::vector<double>::const_iterator getSubMatrConstIter(const std::vector<double>& matr,
    HorAnglePos hor_angle_pos, VertAnglePos vert_angle_pos) {
    if (hor_angle_pos == left) {
        if (vert_angle_pos == up)
            return getLeftUpSubMatrConstIter(matr);
        else
            return getLeftDownSubMatrConstIter(matr);
    } else {
        if (vert_angle_pos == up)
            return getRightUpSubMatrConstIter(matr);
        else
            return getRightDownSubMatrConstIter(matr);
    }
}

std::vector<double>::iterator getLeftUpSubMatrIter(std::vector<double>::iterator matr, int row_num) {
    return matr;
}

std::vector<double>::iterator getLeftDownSubMatrIter(std::vector<double>::iterator matr, int row_num) {
    return matr + ((row_num * row_num) / 2);
}

std::vector<double>::iterator getRightUpSubMatrIter(std::vector<double>::iterator matr, int row_num) {
    return matr + (row_num / 2);
}

std::vector<double>::iterator getRightDownSubMatrIter(std::vector<double>::iterator matr, int row_num) {
    return matr + ((row_num * row_num) / 2) + (row_num / 2);
}


std::vector<double>::iterator getSubMatrIter(std::vector<double>::iterator matr,
    HorAnglePos hor_angle_pos, VertAnglePos vert_angle_pos, int row_num) {
    if (hor_angle_pos == left) {
        if (vert_angle_pos == up)
            return getLeftUpSubMatrIter(matr, row_num);
        else
            return getLeftDownSubMatrIter(matr, row_num);
    } else {
        if (vert_angle_pos == up)
            return getRightUpSubMatrIter(matr, row_num);
        else
            return getRightDownSubMatrIter(matr, row_num);
    }
}

std::vector<double> StrassenMulti(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> c(a.size());

    std::vector<double>::const_iterator a11 = getSubMatrConstIter(a, left, up);
    std::vector<double>::const_iterator a12 = getSubMatrConstIter(a, right, up);
    std::vector<double>::const_iterator a21 = getSubMatrConstIter(a, left, down);
    std::vector<double>::const_iterator a22 = getSubMatrConstIter(a, right, down);

    std::vector<double>::const_iterator b11 = getSubMatrConstIter(b, left, up);
    std::vector<double>::const_iterator b12 = getSubMatrConstIter(b, right, up);
    std::vector<double>::const_iterator b21 = getSubMatrConstIter(b, left, down);
    std::vector<double>::const_iterator b22 = getSubMatrConstIter(b, right, down);

    std::vector<double>::iterator c_it = c.begin();
    int c_row_num = sqrt(c.size());
    std::vector<double>::iterator c11 = getSubMatrIter(c_it, left, up, c_row_num);
    std::vector<double>::iterator c12 = getSubMatrIter(c_it, right, up, c_row_num);
    std::vector<double>::iterator c21 = getSubMatrIter(c_it, left, down, c_row_num);
    std::vector<double>::iterator c22 = getSubMatrIter(c_it, right, down, c_row_num);


    int p_size = a.size() / 4;
    int p_rows_num = sqrt(p_size);
    std::vector<double> m1(p_size);
    std::vector<double> m2(p_size);
    std::vector<double>::iterator m1_it = m1.begin();
    std::vector<double>::iterator m2_it = m2.begin();
    std::vector<double> p1(p_size);
    std::vector<double>::iterator p1_it = p1.begin();
    std::vector<double> p2(p_size);
    std::vector<double>::iterator p2_it = p2.begin();
    std::vector<double> p3(p_size);
    std::vector<double>::iterator p3_it = p3.begin();
    std::vector<double> p4(p_size);
    std::vector<double>::iterator p4_it = p4.begin();
    std::vector<double> p5(p_size);
    std::vector<double>::iterator p5_it = p5.begin();
    std::vector<double> p6(p_size);
    std::vector<double>::iterator p6_it = p6.begin();
    std::vector<double> p7(p_size);
    std::vector<double>::iterator p7_it = p7.begin();

    // !!!P1
    matrSum(a11, a22, m1_it, p_rows_num);
    matrSum(b11, b22, m2_it, p_rows_num);
    matrMulti(m1, m2, p1_it, p_rows_num);
    // P1!!!

    // !!!P2
    matrSum(a21, a22, m1_it, p_rows_num);
    matrMulti(m1, b11, p2_it, p_rows_num);
    // P2!!!

    // !!!P3
    matrDiff(b12, b22, m1_it, p_rows_num);
    matrMulti(a11, m1, p3_it, p_rows_num);
    // P3!!!

    // !!!P4
    matrDiff(b21, b11, m1_it, p_rows_num);
    matrMulti(a22, m1, p4_it, p_rows_num);
    // P4!!!

    // !!!P5
    matrSum(a11, a12, m1_it, p_rows_num);
    matrMulti(m1, b22, p5_it, p_rows_num);
    // P5!!!

    // !!!P6
    matrDiff(a21, a11, m1_it, p_rows_num);
    matrSum(b11, b12, m2_it, p_rows_num);
    matrMulti(m1, m2, p6_it, p_rows_num);
    // P6!!!

    // !!!P7
    matrDiff(a12, a22, m1_it, p_rows_num);
    matrSum(b21, b22, m2_it, p_rows_num);
    matrMulti(m1, m2, p7_it, p_rows_num);
    // P7!!!

    // !!!C11
    matrSum(p1, p4, c11);
    matrDiff(c11, p5);
    matrSum(c11, p7);
    // C11!!!

    // !!!C12
    matrSum(p3, p5, c12);
    // C12!!!

    // !!!C21
    matrSum(p2, p4, c21);
    // C21!!!

    // !!!C22
    matrDiff(p1, p2, c22);
    matrSum(c22, p3);
    matrSum(c22, p6);
    // C22!!!

    return c;
}

void matrSum(std::vector<double>::const_iterator a_it,
    std::vector<double>::const_iterator b_it,
    std::vector<double>::iterator m_it,
    int m_rows_num) {
    int full_matr_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(m_it + i * m_rows_num + j) = *(a_it + i * full_matr_rows_num + j) + *(b_it + i * full_matr_rows_num + j);
        }
    }
}

void matrDiff(std::vector<double>::const_iterator a_it,
    std::vector<double>::const_iterator b_it,
    std::vector<double>::iterator m_it,
    int m_rows_num) {
    int full_matr_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(m_it + i * m_rows_num + j) = *(a_it + i * full_matr_rows_num + j) - *(b_it + i * full_matr_rows_num + j);
        }
    }
}

void matrSum(const std::vector<double>& m1,
    const std::vector<double>& m2,
    std::vector<double>::iterator c_it) {
    int m_rows_num = sqrt(m1.size());
    int c_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(c_it + i * c_rows_num + j) = m1[i * m_rows_num + j] + m2[i * m_rows_num + j];
        }
    }
}

void matrDiff(const std::vector<double>& m1,
    const std::vector<double>& m2,
    std::vector<double>::iterator c_it) {
    int m_rows_num = sqrt(m1.size());
    int c_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(c_it + i * c_rows_num + j) = m1[i * m_rows_num + j] - m2[i * m_rows_num + j];
        }
    }
}

void matrSum(std::vector<double>::iterator c_it,
    const std::vector<double>& m1) {
    int m_rows_num = sqrt(m1.size());
    int c_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(c_it + i * c_rows_num + j) += m1[i * m_rows_num + j];
        }
    }
}

void matrDiff(std::vector<double>::iterator c_it,
    const std::vector<double>& m1) {
    int m_rows_num = sqrt(m1.size());
    int c_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            *(c_it + i * c_rows_num + j) -= m1[i * m_rows_num + j];
        }
    }
}

void matrMulti(std::vector<double>::const_iterator a_it,
    const std::vector<double>& p2,
    std::vector<double>::iterator m_it,
    int m_rows_num) {
    int full_matr_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            for (int k = 0; k < m_rows_num; k++) {
                *(m_it + i * m_rows_num + j) += (*(a_it + i * full_matr_rows_num + k)) * p2[j + k * m_rows_num];
            }
        }
    }
}

void matrMulti(const std::vector<double>& p2,
    std::vector<double>::const_iterator a_it,
    std::vector<double>::iterator m_it,
    int m_rows_num) {
    int full_matr_rows_num = m_rows_num * 2;
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            for (int k = 0; k < m_rows_num; k++) {
                *(m_it + i * m_rows_num + j) += (p2[i * m_rows_num + k] * *(a_it + j + k * full_matr_rows_num));
            }
        }
    }
}

void matrMulti(const std::vector<double>& p1,
    const std::vector<double>& p2,
    std::vector<double>::iterator m_it,
    int m_rows_num) {
    for (int i = 0; i < m_rows_num; i++) {
        for (int j = 0; j < m_rows_num; j++) {
            for (int k = 0; k < m_rows_num; k++) {
                *(m_it + i * m_rows_num + j) += p1[i * m_rows_num + k] * p2[j + k * m_rows_num];
            }
        }
    }
}
