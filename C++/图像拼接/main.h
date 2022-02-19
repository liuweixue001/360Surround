#pragma once
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <time.h>
using namespace std;
using namespace cv;
//���巽λö��
enum DIRECTION { L, R, F, B} dir;
//���ͼ����
void gen_bird(Mat& frame, Size new_size, Matx33d homogr) {
	warpPerspective(frame, frame, homogr, new_size);
}
//ͼ��ȥ����
void undistortion(Mat& frame, Matx33d intrinsic_matrix, Vec4d distortion_coeffs, Size image_size, Matx33d new_matrix) {
	Mat mapx(frame.rows, frame.cols, CV_32FC1, Scalar(0));
	Mat mapy(frame.rows, frame.cols, CV_32FC1, Scalar(0));
	Mat R = Mat::eye(3, 3, CV_32FC1);
	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
		new_matrix, image_size, CV_32FC1, mapx, mapy);
	remap(frame, frame, mapx, mapy, INTER_LINEAR, 0);
}


