// Copyright 2021 Gruzdeva Diana

#include <gtest/gtest.h>

#include <omp.h>
#include <random>
#include <vector>

#include "../../../modules/task_2/gruzdeva_d_fox_mult/fox_mult.h"

TEST(Fox_Multiplication_OMP, DIRECT_AND_FOX_1) {
    std::vector<double> A = getRandomMatrix(100, time(0));

    std::vector<double> B = getRandomMatrix(100, time(0) + 1);

    std::vector<double> C1 = directMultiplication(A, B);

    std::vector<double> C2 = parallelFoxMultiplication(A, B, 4);

    ASSERT_EQ(C1, C2);
}

TEST(Fox_Multiplication_OMP, DIRECT_AND_FOX_2) {
    std::vector<double> A = getRandomMatrix(250, time(0));

    std::vector<double> B = getRandomMatrix(250, time(0) + 1);

    std::vector<double> C1 = directMultiplication(A, B);

    std::vector<double> C2 = parallelFoxMultiplication(A, B, 4);

    ASSERT_EQ(C1, C2);
}

TEST(Fox_Multiplication_OMP, SEQ_FOX_AND_PAR_FOX_1) {
    std::vector<double> A = getRandomMatrix(100, time(0));

    std::vector<double> B = getRandomMatrix(100, time(0) + 1);
    double t1 = omp_get_wtime();
    std::vector<double> C1 = sequentialFoxMultiplication(A, B, 4);
    double t2 = omp_get_wtime();
    std::cout << "Direct: " << t2-t1 << "\n";
    t1 = omp_get_wtime();
    std::vector<double> C2 = parallelFoxMultiplication(A, B, 4);
    t2 = omp_get_wtime();
    std::cout << "OMP: " << t2-t1 << "\n";

    ASSERT_EQ(C1, C2);
}

TEST(Fox_Multiplication_OMP, SEQ_FOX_AND_PAR_FOX_2) {
    std::vector<double> A = getRandomMatrix(500, time(0));

    std::vector<double> B = getRandomMatrix(500, time(0) + 1);
    double t1 = omp_get_wtime();
    std::vector<double> C1 = sequentialFoxMultiplication(A, B, 4);
    double t2 = omp_get_wtime();
    std::cout << "Direct: " << t2-t1 << "\n";
    t1 = omp_get_wtime();
    std::vector<double> C2 = parallelFoxMultiplication(A, B, 4);
    t2 = omp_get_wtime();
    std::cout << "OMP: " << t2-t1 << "\n";

    ASSERT_EQ(C1, C2);
}

TEST(Fox_Multiplication_OMP, DIRECT_AND_FOX_4) {
    std::vector<double> A = getRandomMatrix(100, time(0));

    std::vector<double> B = getRandomMatrix(100, time(0) + 1);

    std::vector<double> C1 = directMultiplication(A, B);

    std::vector<double> C2 = parallelFoxMultiplication(A, B, 4);

    ASSERT_EQ(C1, C2);
}

TEST(Fox_Multiplication_OMP, THROW_DIFFERENT_MATRICES_SIZES) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(5, time(0) + 1);

    ASSERT_ANY_THROW(parallelFoxMultiplication(A, B, 4));
}

TEST(Fox_Multiplication_OMP, THROW_NEGATIVE_THREAD) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(parallelFoxMultiplication(A, B, -2));
}

TEST(Fox_Multiplication_OMP, THROW_WRONG_THREAD_NUMBER) {
    std::vector<double> A = getRandomMatrix(5, time(0));

    std::vector<double> B = getRandomMatrix(5, time(0) + 1);

    ASSERT_ANY_THROW(parallelFoxMultiplication(A, B, 8));
}

TEST(Fox_Multiplication_OMP, THROW_NULL_THREAD_NUMBER) {
    std::vector<double> A = getRandomMatrix(4, time(0));

    std::vector<double> B = getRandomMatrix(4, time(0) + 1);

    ASSERT_ANY_THROW(parallelFoxMultiplication(A, B, 0));
}
