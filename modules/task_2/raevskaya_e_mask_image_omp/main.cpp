// Copyright 2021 Ekaterina Raevskaya

#include <gtest/gtest.h>
#include <omp.h>
#include <ctime>
#include "./mask_image.h"

TEST(mask_image, Test_Mask_image_1) {
    int n = 4, m = 4;
    std::vector <int> a = {
        1, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 0, 1,
        0, 1, 0, 1
    };
    std::vector <int> ans = {
        1, 0, 0, 0,
        0, 6, 6, 0,
        0, 6, 0, 12,
        0, 6, 0, 12
    };
    ASSERT_EQ(mask_dsu(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_2) {
    int n = 4, m = 4;
    std::vector <int> a = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    std::vector <int> ans = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    ASSERT_EQ(mask_dsu(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_3) {
    int n = 5, m = 4;
    std::vector <int> a = {
        1, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 0, 1,
        0, 1, 0, 1,
        1, 0, 1, 0
    };
    std::vector <int> ans = {
        1, 0, 0, 0,
        0, 6, 6, 0,
        0, 6, 0, 12,
        0, 6, 0, 12,
        17, 0, 19, 0
    };
    ASSERT_EQ(mask_dsu(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_4) {
    int n = 5, m = 6;
    std::vector <int> a = {
        1, 0, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 1
    };
    std::vector <int> ans = {
        1, 0, 0, 0, 5, 0,
        0, 8, 8, 0, 5, 0,
        0, 8, 0, 5, 5, 5,
        0, 8, 0, 5, 0, 0,
        25, 0, 27, 0, 29, 29
    };
    ASSERT_EQ(mask_dsu(&a, n, m), ans);
}

TEST(mask_image, Test_Mask_image_5) {
    int n = 5, m = 6;
    std::vector <int> a = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1
    };
    std::vector <int> ans = {
        1, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        25, 0, 27, 27, 27, 27
    };
    ASSERT_EQ(mask_dsu(&a, n, m), ans);
}

/*TEST(mask_image, Test_Mask_image_6) {
    int n = 5500, m = 5500;
    std::vector <int> a(n*m);
    for (int i = 0; i < n*m; i++) {
        a[i] = rand() % 2;
    }

    double start = clock();
    std::vector <int> ans = mask_dsu(&a, n, m);
    double end = clock();

    printf("Works took %f seconds\n", (end-start)/1000.0);

    ASSERT_TRUE(true);
}

TEST(mask_image, Test_Mask_image_7) {
    int n = 5500, m = 5500;
    std::vector <int> a(n*m);
    for (int i = 0; i < n*m; i++) {
        a[i] = rand() % 2;
    }

    double start = omp_get_wtime();
    std::pair<std::vector <int>, int> ans = mask_dsu_omp(&a, n, m);
    double end = omp_get_wtime();

    printf("Work took %f seconds\n", end - start);

    ASSERT_TRUE(true);
}*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
