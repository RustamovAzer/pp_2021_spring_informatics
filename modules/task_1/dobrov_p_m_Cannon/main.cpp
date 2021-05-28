// Copyright 2021 Dobrov Pavel
#include <gtest/gtest.h>
#include <vector>
#include "./m_cannon.h"

TEST(m_cannon, comparison_of_NaiveMulti_and_MatrixMulti) {
    std::vector<double> A = RandomMatrix(16, 19);
    std::vector<double> B = RandomMatrix(19, 16);

    std::vector<double> C1 = NaiveMulti(A, B, 16, 19, 16);
    std::vector<double> C2 = MatrixMulti(A, B, 16, 19, 16);

    for (size_t i = 0; i < C1.size(); i++) {
        ASSERT_DOUBLE_EQ(C1[i], C2[i]);
    }
}

TEST(m_Cannon, matrix_with_negative_size) {
    ASSERT_ANY_THROW(RandomMatrix(-1, -1));
}

TEST(m_Cannon, Cant_execute_with_negative_size_MatrixMulti) {
    std::vector<double> A(1);
    std::vector<double> B(1);
    ASSERT_ANY_THROW(MatrixMulti(A, B, -1, -1, -1));
}

TEST(m_Cannon, Cant_execute_with_negative_size_NaiveMulti) {
    std::vector<double> A(1);
    std::vector<double> B(1);
    ASSERT_ANY_THROW(NaiveMulti(A, B, -1, -1, -1));
}

TEST(m_Cannon, NaiveMulti) {
    std::vector<double> A = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> B = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> C = { 1, 2, 5, 8,
                             0, 1, 2, 5,
                             0, 0, 1, 2,
                             0, 0, 0, 1 };

    std::vector<double> MultiMatrix = NaiveMulti(A, B, 4, 4, 4);
    ASSERT_EQ(MultiMatrix, C);
}

TEST(m_Cannon, MatrixMulti) {
    std::vector<double> A = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> B = { 1, 1, 2, 2,
                             0, 1, 1, 2,
                             0, 0, 1, 1,
                             0, 0, 0, 1 };

    std::vector<double> C = { 1, 2, 5, 8,
                             0, 1, 2, 5,
                             0, 0, 1, 2,
                             0, 0, 0, 1 };

    std::vector<double> MultiMatrix = MatrixMulti(A, B, 4, 4, 4);
    ASSERT_EQ(MultiMatrix, C);
}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
