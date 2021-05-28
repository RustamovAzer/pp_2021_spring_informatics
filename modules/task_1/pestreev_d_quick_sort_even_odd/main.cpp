// Copyright 2021 Pestreev Daniil
#include <gtest/gtest.h>
#include <vector>
#include "./quick_sort_even_odd_merge.h"

TEST(Quick_Sort_Operations, Test_manual_vector) {
    std::vector<int> vec = {-8098, 1009, -160, -1, 179796, 2,
        -1603, 166, -7396, -19, -18348, -15, 659, 7, 82519};
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_random_vector_0) {
    std::vector<int> vec = getRandomVector(0);
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_random_vector_1) {
    std::vector<int> vec = getRandomVector(1);
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_sorted_vector) {
    std::vector<int> vec = getRandomVector(42211);
    std::sort(vec.begin(), vec.end());
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_inverted_sorted_vector42211) {
    std::vector<int> vec = getRandomVector(42211);
    std::sort(vec.begin(), vec.end(), [](int a, int b){
        return a > b;
    });
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_random_vector_42211) {
    std::vector<int> vec = getRandomVector(42211);
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}
TEST(Quick_Sort_Operations, Test_random_vector_3412211) {
    std::vector<int> vec = getRandomVector(3412211);
    std::vector<int> vec_V = quickSortV(vec);
    std::vector<int> vec_stan = vec;
    std::sort(vec_stan.begin(), vec_stan.end());
    ASSERT_EQ(vec_V, vec_stan);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
