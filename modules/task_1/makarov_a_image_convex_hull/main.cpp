// Copyright 2021 Makarov Alexander
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "./image_convex_hull.h"

TEST(Components, Test_Snow) {
    const std::vector<int> image = {
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }

    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Fat_Snow) {
    const std::vector<int> image = {
        0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }   
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_4_Plus) {
    const std::vector<int> image = {
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_4_Squares) {
    const std::vector<int> image = {
        0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }*/

    /*std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Perim_sqres) {
    const std::vector<int> image = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Perim_no_angles) {
    const std::vector<int> image = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
