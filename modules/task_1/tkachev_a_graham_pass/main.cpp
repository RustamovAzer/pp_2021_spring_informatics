// Copyright 2021 Tkachev Alexey
#include <gtest/gtest.h>
#include "../../../modules/task_1/tkachev_a_graham_pass/graham_pass.h"

TEST(test_tkachev_a_graham_pass, test_1_big_size_1000) {
    std::vector<Point> const points =
        getRandomizedVector(-200.0, 200.0, 1000);
    EXPECT_NO_THROW(useGrahamAlgorithm(points));
}

TEST(test_tkachev_a_graham_pass, test_2_size_7_plus_n_minus_int) {
    std::vector<Point> const points = {
        {2, 3}, {1, -5}, {0, 4}, {-2, 4},
        {10, 2}, {30, -3}, {-15, 2}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {1, -15, -2, 0, 10, 30, 1};

    std::vector<double> true_solution_y =
    {-5, 2, 4, 4, 2, -3, -5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_3_size_15_plus_only_int) {
    std::vector<Point> const points = {
        {0, 2}, {1, 5}, {20, 1}, {15, 7},
        {0, 0}, {0, 1}, {0, 2}, {16, 12},
        {12, 16}, {15, 15}, {30, 45}, {0, 4},
        {13, 14}, {15, 16}, {17, 18}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {0, 0, 30, 20, 0};

    std::vector<double> true_solution_y =
    {0, 4, 45, 1, 0};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_4_size_8_minus_only_double) {
    std::vector<Point> const points = {
        {-5.2, -1.1}, {-3.5, -5.9}, {-10.0, -30.5}, {-0.1, -0.9},
        {-10.5, -12.3}, {-6.3, -10.2}, {-15.4, -2.2}, {-28.8, -28.8}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {-10, -28.8, -15.4, -5.2, -0.1, -10};

    std::vector<double> true_solution_y =
    {-30.5, -28.8, -2.2, -1.1, -0.9, -30.5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_5_size_3_plus_minus) {
    std::vector<Point> const points = {
        {0.5, 5.5}, {-10.2, -5.3}, {0.0, 0.0}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {0.5, 0, -10.2, 0.5};

    std::vector<double> true_solution_y =
    {5.5, 0, -5.3, 5.5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_6_size_3_plus_only) {
    std::vector<Point> const points = {
        {10.3, 20.5}, {30.1, 40.8}, {20.3, 50.4}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {10.3, 20.3, 30.1, 10.3};

    std::vector<double> true_solution_y =
    {20.5, 50.4, 40.8, 20.5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_7_size_3_minus_only) {
    std::vector<Point> const points = {
        {-10.3, -1.2}, {-30.2, -12.3}, {-3.1, -5.4}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {-10.3, -3.1, -30.2, -10.3};

    std::vector<double> true_solution_y =
    {-1.2, -5.4, -12.3, -1.2};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_8_size_2) {
    std::vector<Point> const points = {
        {-1.5, 1.5}, {2.5, -2.5}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {2.5, -1.5};

    std::vector<double> true_solution_y =
    {-2.5, 1.5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}

TEST(test_tkachev_a_graham_pass, test_9_size_1) {
    std::vector<Point> const points = {
        {-1.5, 1.5}
    };
    std::stack<Point> graham = useGrahamAlgorithm(points);
    std::vector<double> graham_x(graham.size(), 0.0);
    std::vector<double> graham_y(graham.size(), 0.0);

    for (uint32_t i = 0; i < graham_x.size(); i++) {
        graham_x[i] = graham.top().x;
        graham_y[i] = graham.top().y;
        graham.pop();
    }

    std::vector<double> true_solution_x =
    {-1.5};

    std::vector<double> true_solution_y =
    {1.5};

    ASSERT_EQ(graham_x, true_solution_x);
    ASSERT_EQ(graham_y, true_solution_y);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
