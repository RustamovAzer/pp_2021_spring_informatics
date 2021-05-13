// Copyright 2021 Rustamov Azer

#include <gtest/gtest.h>
#include <iostream>
#include <tbb/tick_count.h>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../../../modules/task_3/rustamov_a_histogram_stretch/histogram_stretch.h"

using namespace cv;

TEST(Histogram_Stretch, Show_Stretch_Algorithm) {
    int w = 10000, h = 1000;
    tbb::tick_count time_start = tbb::tick_count::now();

    Matrix image = generate_random_image_tbb(w, h, 10, 100);
    tbb::tick_count time_generate_image = tbb::tick_count::now();

    Matrix histogram = make_histogram_tbb(image, w, h);

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
    cv::imshow("input_histogram", hist_image);
    waitKey(0);
    tbb::tick_count time_start_alg_seq = tbb::tick_count::now();

    Matrix result_seq = histogram_sretch_algorithm(image, w, h);
    tbb::tick_count time_finish_alg_seq = tbb::tick_count::now();

    Matrix hist_res_seq = make_histogram(result_seq, w, h);
    tbb::tick_count time_make_hist_seq = tbb::tick_count::now();

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
    cv::imshow("stretched_histogram_seq", res_hist_seq_image);
    waitKey(0);
    tbb::tick_count time_start_alg_tbb = tbb::tick_count::now();

    Matrix result_tbb = histogram_sretch_algorithm_tbb(image, w, h);
    tbb::tick_count time_finish_alg_tbb = tbb::tick_count::now();

    Matrix hist_res_tbb = make_histogram_tbb(result_tbb, w, h);
    tbb::tick_count time_make_hist_tbb = tbb::tick_count::now();

    Mat res_tbb_img = Mat(img.rows, img.cols, CV_8UC1);
    memcpy(res_tbb_img.data, result_tbb.data(), result_tbb.size()*sizeof(unsigned char));
    namedWindow("result_image_tbb", WINDOW_AUTOSIZE);
    imshow("result_image_tbb", res_tbb_img);
    waitKey(0);

    Mat3b res_hist_tbb_image = Mat3b::zeros(256, 256);
    for(int b = 0; b < 256; b++) {
        cv::line
            ( res_hist_tbb_image
            , cv::Point(b, 256 - hist_res_tbb[b]), cv::Point(b, 256)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow("stretched_histogram_tbb", res_hist_tbb_image);
    waitKey(0);

    cv::destroyAllWindows();

    std::cout << "FOR " << h * w << " PIXEL IMAGE" << std::endl;
    std::cout << "GENERATE IMAGE: " <<
        (time_generate_image - time_start).seconds() << std::endl;
    double total_time_seq =
        (time_finish_alg_seq - time_start_alg_seq).seconds();
    std::cout << "ALGORITHM SEQ: " << total_time_seq << std::endl;
    double total_time_tbb =
        (time_finish_alg_tbb - time_start_alg_tbb).seconds();
    std::cout << "ALGORITHM TBB: " << total_time_tbb << std::endl;
    std::cout << "ACCELERATION: " << total_time_seq / total_time_tbb << std::endl;


    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result_seq[i], result_tbb[i]);
    }
}
/*
TEST(Histogram_Stretch, Incorrect_Image) {
    int w = 0, h = 10;
    ASSERT_ANY_THROW(generate_random_image(w, h));
}

TEST(Histogram_Stretch, Correct_Stretch_Histogram_Parallel) {
    int min_y, max_y;
    int w = 500, h = 500;
    Matrix str_hist_seq(256), str_hist_tbb(256);

    Matrix image = generate_random_image_tbb(w, h);
    Matrix hist_tbb = make_histogram_tbb(image, w, h);
    min_y = get_min_y(hist_tbb);
    max_y = get_max_y(hist_tbb);

    Matrix res_seq = increase_contrast(image, w, h, min_y, max_y);
    Matrix res_tbb = increase_contrast_tbb(image, w, h, min_y, max_y);

    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(res_seq[i], res_tbb[i]);
    }
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
    Matrix result = histogram_sretch_algorithm_tbb(image, w, h);
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
    Matrix result = histogram_sretch_algorithm_tbb(image, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result[i], exp_result[i]);
    }
}

TEST(Histogram_Stretch, Correct_Stretching_500x500) {
    int w = 500, h = 500;
    Matrix image = generate_random_image(w, h);
    Matrix result_seq = histogram_sretch_algorithm(image, w, h);
    Matrix result_tbb = histogram_sretch_algorithm_tbb(image, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result_seq[i], result_tbb[i]);
    }
}

TEST(Histogram_Stretch, Cannot_Stretch_Twice) {
    int w = 100, h = 100;
    Matrix image = generate_random_image(w, h);
    Matrix result = histogram_sretch_algorithm_tbb(image, w, h);
    Matrix second_result = histogram_sretch_algorithm_tbb(result, w, h);
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result[i], second_result[i]);
    }
}
*/
/*
TEST(Histogram_Stretch, Compare_Preprocessing_Seq_And_TBB) {
    int w = 1000, h = 1000;
    int min_s, max_s, min_t, max_t;

    tbb::tick_count  time_n1 = tbb::tick_count::now();
    Matrix image_seq = generate_random_image(w, h);

    tbb::tick_count  time0 = tbb::tick_count::now();
    Matrix image = generate_random_image_tbb(w, h); 

    tbb::tick_count  time1 = tbb::tick_count::now();
    Matrix hist_seq = make_histogram(image, w, h);

    tbb::tick_count  time2 = tbb::tick_count::now();
    Matrix hist_tbb = make_histogram_tbb(image, w, h);  

    tbb::tick_count  time3 = tbb::tick_count::now();
    min_s = get_min_y(hist_seq);
    max_s = get_max_y(hist_seq);

    tbb::tick_count  time4 = tbb::tick_count::now();
    min_t = better_get_min_y_tbb(image, h, w);
    max_t = better_get_max_y_tbb(image, h, w);

    tbb::tick_count  time5 = tbb::tick_count::now();

    std::cout << "FOR " << h * w << " PIXEL IMAGE" << std::endl;
    std::cout << "GENERATE IMAGE" << std::endl;
    std::cout << "SEQ: " << (time0 - time_n1).seconds() << std::endl;
    std::cout << "TBB: " << (time1 - time0).seconds() << std::endl;
    std::cout << "ACCELERATION: " <<
        (time0 - time_n1).seconds() / (time1 - time0).seconds() << std::endl;

    std::cout << "MAKE HISTOGRAM" << std::endl;
    std::cout << "SEQ: " << (time2 - time1).seconds() << std::endl;
    std::cout << "TBB: " << (time3 - time2).seconds() << std::endl;
    std::cout << "ACCELERATION: " <<
        (time2 - time1).seconds() / (time3 - time2).seconds() << std::endl;
    std::cout << "FIND MIN MAX Y" << std::endl;
    std::cout << "SEQ: " << (time4 - time3).seconds() << std::endl;
    std::cout << "TBB NEW: " <<
        (time5 - time4).seconds() << std::endl << std::endl;

    std::cout << "TOTAL TO GET MIN MAX Y" << std::endl;
    double total_time_seq =
        (time2 - time1).seconds() + (time4 - time3).seconds();
    std::cout << "SEQ: " << total_time_seq << std::endl;
    double total_time_tbb_old =
        (time3 - time2).seconds() + (time4 - time3).seconds();
    std::cout << "TBB OLD: " << total_time_tbb_old << std::endl;
    std::cout << "ACCELERATION OLD: " << 
        total_time_seq / total_time_tbb_old << std::endl;
    double total_time_tbb_new = (time5 - time4).seconds();
    std::cout << "TBB NEW: " << total_time_tbb_new << std::endl;
    std::cout << "ACCELERATION NEW: " << 
        total_time_seq / total_time_tbb_new << std::endl;

    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(hist_seq[i], hist_tbb[i]);
    }
    ASSERT_EQ(min_s, min_t);
    ASSERT_EQ(max_s, max_t);
}

TEST(Histogram_Stretch, Compare_Algorithm_Seq_And_TBB) {
    int w = 1000, h = 1000;
    tbb::tick_count  time0 = tbb::tick_count::now();
    Matrix image = generate_random_image_tbb(w, h);

    tbb::tick_count  time1 = tbb::tick_count::now();
    Matrix result_seq = histogram_sretch_algorithm(image, w, h);

    tbb::tick_count  time2 = tbb::tick_count::now();
    Matrix result_tbb = histogram_sretch_algorithm_tbb(image, w, h);

    tbb::tick_count  time3 = tbb::tick_count::now();

    std::cout << "FOR " << h * w << " PIXEL IMAGE" << std::endl;
    std::cout << "GENERATE IMAGE:" << (time1 - time0).seconds() << std::endl;
    std::cout << "SEQ: " << (time2 - time1).seconds() << std::endl;
    std::cout << "TBB: " << (time3 - time2).seconds() << std::endl;
    std::cout << "ACCELERATION: " <<
        (time2 - time1).seconds() / (time3 - time2).seconds() << std::endl;
    for (int i = 0; i < h * w; i++) {
        ASSERT_EQ(result_seq[i], result_tbb[i]);
    }
}
*/
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
