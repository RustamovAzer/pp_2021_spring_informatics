// Copyright 2021 Rustamov Azer

#include <gtest/gtest.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../modules/task_2/rustamov_a_histogram_stretch/histogram_stretch.h"

using namespace cv;

TEST(Histogram_Stretch, Correct_Histogram_stretch_Algorithm) {
    int w = 1000, h = 1000;
    Matrix image = generate_random_image_parallel(w, h, 10, 100);
    Matrix histogram = make_histogram_parallel(image, w, h);

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

    Matrix result_seq = histogram_sretch_algorithm(image, w, h);
    Matrix hist_res_seq = make_histogram(result_seq, w, h);

    Mat res_seq_img = Mat(img.rows, img.cols, CV_8UC1);
    memcpy(res_seq_img.data, result_seq.data(), result_seq.size()*sizeof(unsigned char));
    namedWindow("result_image_seq", WINDOW_AUTOSIZE);
    imshow("result_image_seq", res_seq_img);
    waitKey(0);


    Mat3b res_hist_seq_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( res_hist_seq_image
            , cv::Point(b, 256 - hist_res_seq[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("STRETCHED_HISTOGRAM_SEQ", res_hist_seq_image);
    waitKey(0);

    Matrix result_omp = histogram_sretch_algorithm_parallel(image, w, h);
    Matrix hist_res_omp = make_histogram(result_omp, w, h);

    Mat res_omp_img = Mat(img.rows, img.cols, CV_8UC1);
    memcpy(res_omp_img.data, result_omp.data(), result_omp.size()*sizeof(unsigned char));
    namedWindow("result_image_omp", WINDOW_AUTOSIZE);
    imshow("result_image_omp", res_omp_img);
    waitKey(0);

    Mat3b res_hist_omp_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( res_hist_omp_image
            , cv::Point(b, 256 - hist_res_omp[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("STRETCHED_HISTOGRAM_OMP", res_hist_omp_image);
    waitKey(0);

    cv::destroyAllWindows();

    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result_seq[i], result_omp[i]);
    }
}

/*
TEST(Histogram_Stretch, Correct_Make_Histogram_Parallel) {
    int w = 500, h = 500;

    Matrix image = generate_random_image_parallel(w, h);
    Matrix hist_seq = make_histogram(image, w, h);
    Matrix hist_omp = make_histogram_parallel(image, w, h);

    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(hist_seq[i], hist_omp[i]);
    }
}

TEST(Histogram_Stretch, Correct_Stretch_histogram_Parallel) {
    int min_y, max_y;
    int w = 500, h = 500;
    Matrix str_hist_seq(256), str_hist_omp(256);

    Matrix image = generate_random_image_parallel(w, h);
    Matrix hist_omp = make_histogram_parallel(image, w, h);
    min_y = get_min_y(hist_omp);
    max_y = get_max_y(hist_omp);

    Matrix res_seq = increase_contrast(image, w, h, min_y, max_y);
    Matrix res_omp = increase_contrast_parallel(image, w, h, min_y, max_y);

    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(res_seq[i], res_omp[i]);
    }
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
*/
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
