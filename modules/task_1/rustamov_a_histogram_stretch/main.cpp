// Copyright 2021 Rustamov Azer

#include <gtest/gtest.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../modules/task_1/rustamov_a_histogram_stretch/histogram_stretch.h"

using namespace cv;


TEST(Histogram_Stretch, Show_Image) {
    Mat image = 
        imread("C:\\Users\\Zerus\\Documents\\pp_2021_spring_informatics\\modules\\task_1\\rustamov_a_histogram_stretch\\test.png",
            IMREAD_GRAYSCALE);
    namedWindow("original_image", WINDOW_AUTOSIZE);
    imshow("original_image", image);
    waitKey(0);
    
    int w = image.cols;
    int h = image.rows;
    Mat flat = image.reshape(1, image.total());
    Matrix histogram = make_histogram(flat, w, h);

    Mat3b hist_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( hist_image
            , cv::Point(b, 256 - histogram[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("INPUT HISTOGRAM", hist_image);
    waitKey(0);

    Matrix result = histogram_sretch_algorithm(flat, w, h);

    Mat m = Mat(image.rows, image.cols, CV_8UC1);
    memcpy(m.data, result.data(), result.size()*sizeof(unsigned char));
    namedWindow("result_image", WINDOW_AUTOSIZE);
    imshow("result_image", m);
    waitKey(0);

    Matrix str_histogram = stretch_histogram(histogram, get_min_y(histogram), get_max_y(histogram));

    Mat3b res_hist_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( res_hist_image
            , cv::Point(b, 256 - str_histogram[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("STRETCHED_HISTOGRAM", res_hist_image);
    waitKey(0);

    cv::destroyAllWindows();

    SUCCEED();
}

TEST(Histogram_Stretch, Show_Stretch_Random_Image_800x450) {
    int w = 800, h = 450;
    Matrix image = generate_random_image(w, h, 10, 100);
    Matrix histogram = make_histogram(image, w, h);

    Mat img = Mat(h, w, CV_8UC1);
    memcpy(img.data, image.data(), image.size()*sizeof(unsigned char));
    namedWindow("generated_image", WINDOW_AUTOSIZE);
    imshow("generated_image", img);
    waitKey(0);

    Mat3b hist_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( hist_image
            , cv::Point(b, 256 - histogram[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("INPUT HISTOGRAM", hist_image);
    waitKey(0);

    Matrix result = histogram_sretch_algorithm(image, w, h);
    Matrix str_histogram = stretch_histogram(histogram, get_min_y(histogram), get_max_y(histogram));

    Mat res = Mat(h, w, CV_8UC1);
    memcpy(res.data, result.data(), result.size()*sizeof(unsigned char));
    namedWindow("result_image", WINDOW_AUTOSIZE);
    imshow("result_image", res);
    waitKey(0);

    Mat3b res_hist_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( res_hist_image
            , cv::Point(b, 256 - str_histogram[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("STRETCHED_HISTOGRAM", res_hist_image);
    waitKey(0);

    cv::destroyAllWindows();

    SUCCEED();
}

TEST(Histogram_Stretch, Incorrect_Image) {
    int w = 0, h = 10;
    ASSERT_ANY_THROW(generate_random_image(w, h));
}

TEST(Histogram_Stretch, Histogram_Of_Result_Equal_To_Stretched_Histogram) {
    int w = 100, h = 100;
    Matrix image = generate_random_image(w, h);
    Matrix histogram = make_histogram(image, w, h);
    int min_y, max_y;
    min_y = get_min_y(histogram);
    max_y = get_max_y(histogram);
    Matrix str_histogram = stretch_histogram(histogram, min_y, max_y);
    Matrix result = histogram_sretch_algorithm(image, w, h);
    Matrix res_histogram = make_histogram(result, w, h);
    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(str_histogram[i], res_histogram[i]);
    }
}

TEST(Histogram_Stretch, Correct_Stretching_2x2) {
    int w = 2, h = 2;
    Matrix image = { 0, 1,
                     1, 0 };
    Matrix exp_result = { 0, 255,
                         255, 0 };
    Matrix histogram = make_histogram(image, w, h);
    Matrix result = histogram_sretch_algorithm(image, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result[i], exp_result[i]);
    }
}

TEST(Histogram_Stretch, Correct_Stretching_10x10) {
    int w = 10, h = 10;
    Matrix image = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, };
    Matrix exp_result = { 0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255,
                          0, 28, 56, 85, 113, 141, 170, 198, 226, 255, };
    Matrix histogram = make_histogram(image, w, h);
    Matrix result = histogram_sretch_algorithm(image, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result[i], exp_result[i]);
    }
}


TEST(Histogram_Stretch, Cannot_Stretch_Twice) {
    int w = 100, h = 100;
    Matrix image = generate_random_image(w, h);
    Matrix result = histogram_sretch_algorithm(image, w, h);
    Matrix second_result = histogram_sretch_algorithm(result, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result[i], second_result[i]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
