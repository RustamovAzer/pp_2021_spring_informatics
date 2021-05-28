// Copyright 2021 Bessolytsin Sergay
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>

#include "../../task_1/bessolitsyn_s_quick_sort/quick_sort.h"

TEST(Quick_Sort, Test_INT_one_element_sort_no_throw) {
    const int size = 1;
    std::vector<int> vec1, vec2;
    vec1 = vec2 = getRandomVector(size, 0);
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

TEST(Quick_Sort, Test_INT_1000_elements_sort) {
    const int size = 1000;
    std::vector<int> vec1, vec2;
    vec1 = vec2 = getRandomVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

TEST(Quick_Sort, Test_INT_1007_elements_sort) {
    const int size = 1007;
    std::vector<int> vec1, vec2;
    vec1 = vec2 = getRandomVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

/*TEST(Quick_Sort, Test_INT_1007007_elements_sort) {
    const int size = 1007007;
    std::vector<int> vec1, vec2;
    vec1 = vec2 = getRandomVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}*/

TEST(Quick_Sort, Test_DOUBLE_one_element_sort_no_throw) {
    const int size = 1;
    std::vector<double> vec1, vec2;
    vec1 = vec2 = getRandomDoubleVector(size, 0);
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

TEST(Quick_Sort, Test_DOUBLE_1000_elements_sort) {
    const int size = 1000;
    std::vector<double> vec1, vec2;
    vec1 = vec2 = getRandomDoubleVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

TEST(Quick_Sort, Test_DOUBLE_1007_elements_sort) {
    const int size = 1007;
    std::vector<double> vec1, vec2;
    vec1 = vec2 = getRandomDoubleVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}

/*TEST(Quick_Sort, Test_DOUBLE_1007007_elements_sort) {
    const int size = 1007007;
    std::vector<double> vec1, vec2;
    vec1 = vec2 = getRandomDoubleVector(size);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_NO_THROW(quick_sort(vec2.data(), size));
    ASSERT_EQ(vec1, vec2);
}*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
