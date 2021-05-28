// Copyright 2021 Chistov Vladimir

#include <omp.h>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "../../../modules/task_2/chistov_v_gauss_block/gauss_block.h"

TEST(Parallel_Count_Sentences_MPI, Image500x500) {
    omp_set_num_threads(4);
    int n = 500;
    double resS, resP;
    double time1, time2, time3, time4;
    std::vector<double> mas(n * n), res(n * n), res1(n * n);
    mas = GenRandVec(n * n);
    time3 = omp_get_wtime();
    res = Gauss_Sequential(mas, n, n);
    time4 = omp_get_wtime();
    resS = time4 - time3;
    time1 = omp_get_wtime();
    res1 = Gauss_Parallel(mas, n, n);
    time2 = omp_get_wtime();
    resP = time2 - time1;
    std::cout << "Time Parallel = " << resP << std::endl <<
        "Time Sequential = " << resS << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Image1920x1080) {
    omp_set_num_threads(5);
    int n = 1920;
    int m = 1080;
    double resS, resP;
    double time1, time2, time3, time4;
    std::vector<double> mas(n * m), res(n * m), res1(n * m);
    mas = GenRandVec(n * n);
    time3 = omp_get_wtime();
    res = Gauss_Sequential(mas, n, m);
    time4 = omp_get_wtime();
    resS = time4 - time3;
    time1 = omp_get_wtime();
    res1 = Gauss_Parallel(mas, n, m);
    time2 = omp_get_wtime();
    resP = time2 - time1;
    std::cout << "Time Parallel = " << resP << std::endl <<
        "Time Sequential = " << resS << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Image2000x2000) {
    omp_set_num_threads(6);
    int n = 2000;
    double resS, resP;
    double time1, time2, time3, time4;
    std::vector<double> mas(n * n), res(n * n), res1(n * n);
    mas = GenRandVec(n * n);
    time3 = omp_get_wtime();
    res = Gauss_Sequential(mas, n, n);
    time4 = omp_get_wtime();
    resS = time4 - time3;
    time1 = omp_get_wtime();
    res1 = Gauss_Parallel(mas, n, n);
    time2 = omp_get_wtime();
    resP = time2 - time1;
    std::cout << "Time Parallel = " << resP << std::endl <<
        "Time Sequential = " << resS << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Image5000x2500) {
    omp_set_num_threads(7);
    int n = 2500;
    int m = 5000;
    double resS, resP;
    double time1, time2, time3, time4;
    std::vector<double> mas(m * n), res(m * n), res1(m * n);
    mas = GenRandVec(m * n);
    time3 = omp_get_wtime();
    res = Gauss_Sequential(mas, m, n);
    time4 = omp_get_wtime();
    resS = time4 - time3;
    time1 = omp_get_wtime();
    res1 = Gauss_Parallel(mas, m, n);
    time2 = omp_get_wtime();
    resP = time2 - time1;
    std::cout << "Time Parallel = " << resP << std::endl <<
        "Time Sequential = " << resS << std::endl << std::endl;
    ASSERT_NO_THROW();
}

TEST(Parallel_Count_Sentences_MPI, Image10000x10000) {
    omp_set_num_threads(8);
    int n = 10000;
    double resS, resP;
    double time1, time2, time3, time4;
    std::vector<double> mas(n * n), res(n * n), res1(n * n);
    mas = GenRandVec(n * n);
    time3 = omp_get_wtime();
    res = Gauss_Sequential(mas, n, n);
    time4 = omp_get_wtime();
    resS = time4 - time3;
    time1 = omp_get_wtime();
    res1 = Gauss_Parallel(mas, n, n);
    time2 = omp_get_wtime();
    resP = time2 - time1;
    std::cout << "Time Parallel = " << resP << std::endl <<
        "Time Sequential = " << resS << std::endl << std::endl;
    ASSERT_NO_THROW();
}
