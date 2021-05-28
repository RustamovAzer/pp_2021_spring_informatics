// Copyright 2020 Chistov Vladimir

#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <ctime>
#include "../../../modules/task_1/chistov_v_gauss_block/gauss_block.h"

TEST(Parallel_Count_Sentences_MPI, Param_3) {
    int a = 1920;
    int b = 1080;
    std::vector<double> mas(a * b), res(a * b), res1(a * b);
    mas = GenRandVec(a * b);
    double time1 = clock();
    res1 = Gauss_Parallel(mas, a, b, 3);
    double time2 = clock();
    double resP = time2 - time1;

    double time3 = clock();
    res = Gauss_Sequential(mas, a, b);
    double time4 = clock();
    double resS = time4 - time3;
    std::cout << "Time Parallel = " << resP / 1000 << std::endl <<
        "Time Sequential = " << resS / 1000 << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Param_8) {
    int a = 1920;
    int b = 1080;
    std::vector<double> mas(a * b), res(a * b), res1(a * b);
    mas = GenRandVec(a * b);
    double time1 = clock();
    res1 = Gauss_Parallel(mas, a, b, 3);
    double time2 = clock();
    double resP = time2 - time1;

    double time3 = clock();
    res = Gauss_Sequential(mas, a, b);
    double time4 = clock();
    double resS = time4 - time3;
    std::cout << "Time Parallel = " << resP / 1000 << std::endl <<
        "Time Sequential = " << resS / 1000 << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Param_12) {
    int a = 1920;
    int b = 1080;
    std::vector<double> mas(a * b), res(a * b), res1(a * b);
    mas = GenRandVec(a * b);
    double time1 = clock();
    res1 = Gauss_Parallel(mas, a, b, 3);
    double time2 = clock();
    double resP = time2 - time1;

    double time3 = clock();
    res = Gauss_Sequential(mas, a, b);
    double time4 = clock();
    double resS = time4 - time3;
    std::cout << "Time Parallel = " << resP / 1000 << std::endl <<
        "Time Sequential = " << resS / 1000 << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Param_19) {
    int a = 1920;
    int b = 1080;
    std::vector<double> mas(a * b), res(a * b), res1(a * b);
    mas = GenRandVec(a * b);
    double time1 = clock();
    res1 = Gauss_Parallel(mas, a, b, 3);
    double time2 = clock();
    double resP = time2 - time1;

    double time3 = clock();
    res = Gauss_Sequential(mas, a, b);
    double time4 = clock();
    double resS = time4 - time3;
    std::cout << "Time Parallel = " << resP / 1000 << std::endl <<
        "Time Sequential = " << resS / 1000 << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Param_30) {
    int a = 1920;
    int b = 1080;
    std::vector<double> mas(a * b), res(a * b), res1(a * b);
    mas = GenRandVec(a * b);
    double time1 = clock();
    res1 = Gauss_Parallel(mas, a, b, 3);
    double time2 = clock();
    double resP = time2 - time1;

    double time3 = clock();
    res = Gauss_Sequential(mas, a, b);
    double time4 = clock();
    double resS = time4 - time3;
    std::cout << "Time Parallel = " << resP / 1000 << std::endl <<
        "Time Sequential = " << resS / 1000 << std::endl << std::endl;
    ASSERT_NO_THROW();
}
