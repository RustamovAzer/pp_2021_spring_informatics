// Copyright 2021 Luckianchencko Ivan
#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "./obolochka.h"

TEST(obolocha, test1) {
    int size = 5;
    mat a(size, std::vector<int>(size));
    a = get_rand_mat(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    double t1, t2, pt1, pt2, tres, ptres;
    t1 = omp_get_wtime();
    std::vector<point> res = Jarvis_alg(a, size);
    t2 = omp_get_wtime();
    tres = t2 - t1;
    omp_set_num_threads(1);
    pt1 = omp_get_wtime();
    std::vector<point> par_res = Paral_Jarvis_alg(a, size);
    pt2 = omp_get_wtime();
    ptres = pt2 - pt1;
    std::cout << " Seq_Time = " << tres << std::endl <<
    " Paral_Time = " << ptres << std::endl;
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, par_res[i].x);
        ASSERT_EQ(res[i].y, par_res[i].y);
    }
}
TEST(obolocha, test2) {
    int size = 10;
    mat a(size, std::vector<int>(size));
    a = get_rand_mat(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    double t1, t2, pt1, pt2, tres, ptres;
    t1 = omp_get_wtime();
    std::vector<point> res = Jarvis_alg(a, size);
    t2 = omp_get_wtime();
    tres = t2 - t1;
    omp_set_num_threads(2);
    pt1 = omp_get_wtime();
    std::vector<point> par_res = Paral_Jarvis_alg(a, size);
    pt2 = omp_get_wtime();
    ptres = pt2 - pt1;
    std::cout << " Seq_Time = " << tres << std::endl <<
    " Paral_Time = " << ptres << std::endl;
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, par_res[i].x);
        ASSERT_EQ(res[i].y, par_res[i].y);
    }
}
TEST(obolocha, test3) {
    int size = 30;
    mat a(size, std::vector<int>(size));
    a = get_rand_mat(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    double t1, t2, pt1, pt2, tres, ptres;
    t1 = omp_get_wtime();
    std::vector<point> res = Jarvis_alg(a, size);
    t2 = omp_get_wtime();
    tres = t2 - t1;
    omp_set_num_threads(3);
    pt1 = omp_get_wtime();
    std::vector<point> par_res = Paral_Jarvis_alg(a, size);
    pt2 = omp_get_wtime();
    ptres = pt2 - pt1;
    std::cout << " Seq_Time = " << tres << std::endl <<
    " Paral_Time = " << ptres << std::endl;
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, par_res[i].x);
        ASSERT_EQ(res[i].y, par_res[i].y);
    }
}
TEST(obolocha, test4) {
    int size = 40;
    mat a(size, std::vector<int>(size));
    a = get_rand_mat(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    double t1, t2, pt1, pt2, tres, ptres;
    t1 = omp_get_wtime();
    std::vector<point> res = Jarvis_alg(a, size);
    t2 = omp_get_wtime();
    tres = t2 - t1;
    omp_set_num_threads(4);
    pt1 = omp_get_wtime();
    std::vector<point> par_res = Paral_Jarvis_alg(a, size);
    pt2 = omp_get_wtime();
    ptres = pt2 - pt1;
    std::cout << " Seq_Time = " << tres << std::endl <<
    " Paral_Time = " << ptres << std::endl;
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, par_res[i].x);
        ASSERT_EQ(res[i].y, par_res[i].y);
    }
}
TEST(obolocha, testsize) {
    int size = 100;
    mat a(size, std::vector<int>(size));
    a = get_rand_mat(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    double t1, t2, pt1, pt2, tres, ptres;
    t1 = omp_get_wtime();
    std::vector<point> res = Jarvis_alg(a, size);
    t2 = omp_get_wtime();
    tres = t2 - t1;
    omp_set_num_threads(5);
    pt1 = omp_get_wtime();
    std::vector<point> par_res = Paral_Jarvis_alg(a, size);
    pt2 = omp_get_wtime();
    ptres = pt2 - pt1;
    std::cout << " Seq_Time = " << tres << std::endl <<
    " Paral_Time = " << ptres << std::endl;
    int count = res.size();
    for (int i = 0; i < count; i++) {
        std::cout << res[i].x << "," << res[i].y << std::endl;
        ASSERT_EQ(res[i].x, par_res[i].x);
        ASSERT_EQ(res[i].y, par_res[i].y);
    }
}


