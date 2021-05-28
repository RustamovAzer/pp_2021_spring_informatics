// Copyright 2021 Emelkhovskaya Ekaterina

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "./strassen.h"

TEST(Sequential, small_matrix_4) {
    int size = 4;
    std::vector<double> a(size * size);
    std::vector<double> b(size * size);
    std::vector<double> c(size * size);
    std::vector<double> real_c(size * size);

    for (int i = 0; i < size * size; i++) {
        a[i] = i + 1;
        b[i] = size * size - i;
    }
    StrassenFunc(a.data(), b.data(), c.data(), size);
    seqMult(a.data(), b.data(), real_c.data(), size);
    ASSERT_EQ(real_c, c);
}

TEST(Sequential, matrix_128) {
    int size = 128;
    std::vector<double> a(size * size);
    std::vector<double> b(size * size);
    std::vector<double> c(size * size);
    std::vector<double> real_c(size * size);

    for (unsigned int i = 0; i < size * size; i++) {
        a[i] = i + 1;
        b[i] = size * size - i;
    }
    StrassenFunc(a.data(), b.data(), c.data(), size);
    seqMult(a.data(), b.data(), real_c.data(), size);
    ASSERT_EQ(real_c, c);
}

TEST(Sequential, matrix_32) {
    int size = 32;
    std::vector<double> a(size * size);
    std::vector<double> b(size * size);
    std::vector<double> c(size * size);
    std::vector<double> real_c(size * size);

    for (unsigned int i = 0; i < size * size; i++) {
        a[i] = i + 1;
        b[i] = size * size - i;
    }
    StrassenFunc(a.data(), b.data(), c.data(), size);
    seqMult(a.data(), b.data(), real_c.data(), size);
    ASSERT_EQ(real_c, c);
}

TEST(Sequential, small_matrix_5_not_2) {
    int size = 5;
    std::vector<double> a(size * size);
    std::vector<double> b(size * size);
    std::vector<double> c(size * size);

    for (unsigned int i = 0; i < size * size; i++) {
        a[i] = i + 1;
        b[i] = size * size - i;
    }
    ASSERT_THROW(StrassenFunc(a.data(), b.data(), c.data(), size),
    std::invalid_argument);
}

TEST(Sequential, matrix_64) {
    int size = 64;
    std::vector<double> a(size * size);
    std::vector<double> b(size * size);
    std::vector<double> c(size * size);
    std::vector<double> real_c(size * size);
    for (unsigned int i = 0; i < size * size; i++) {
        a[i] = i + 1;
        b[i] = size * size - i;
    }
    StrassenFunc(a.data(), b.data(), c.data(), size);
    seqMult(a.data(), b.data(), real_c.data(), size);
    ASSERT_EQ(real_c, c);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
