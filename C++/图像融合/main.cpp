/*
图像融合
5、由于俯视变换后的图像存在较大误差，未进行图像融合
6、采用裁剪图片，进行拼接、融合
*/


#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <time.h>
using namespace std;
using namespace cv;
// -------------------------------------------定义方位枚举-------------------------------------------
enum DIRECTION { L, R, F, B } dir;
// ---------------------------------------------读取视频---------------------------------------------
VideoCapture capture_left("videos/left.avi");
VideoCapture capture_right("videos/right.avi");
VideoCapture capture_front("videos/front.avi");
VideoCapture capture_back("videos/back.avi");

// -------------------------------------定义图片在背景中的放置位置-----------------------------------
Rect G_LF(0, 0, 240, 120);
Rect G_LB(0, 360, 240, 120);
Rect G_RF(480, 0, 240, 120);
Rect G_RB(480, 360, 240, 120);
Rect G_LEFT(0, 120, 240, 240);
Rect G_RIGHT(480, 120, 240, 240);
Rect G_FRONT(240, 0, 240, 120);
Rect G_BACK(240, 360, 240, 120);
Rect G_CAR(240, 120, 240, 240);
// --------------------------------------------定义裁剪区域------------------------------------------
// 1、主方位
Rect LEFT(0, 120, 240, 240);
Rect RIGHT(0, 120, 240, 240);
Rect FRONT(240, 0, 240, 120);
Rect BACK(240, 0, 240, 120);
// 2、角方位
Rect LEFT_LF(0, 0, 240, 120); //左图 左上
Rect FRONT_LF(0, 0, 240, 120);//前图 左上
Rect RIGHT_RF(0, 0, 240, 120);//右图 右上
Rect FRONT_RF(480, 0, 240, 120);// 前图 右上
Rect LEFT_LB(0, 360, 240, 120);//左图 左后
Rect BACK_LB(0, 0, 240, 120);//后图 左后
Rect RIGHT_RB(0, 360, 240, 120);//右图 右后
Rect BACK_RB(480, 0, 240, 120);//后图 右后

// -------------------------------------------定义侧面图像-------------------------------------------
Mat frame_left, frame_right, frame_front, frame_back;
// -------------------------------------------定义角点图片-------------------------------------------
// 1、左图：左前、左后
Mat frame_llf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_llb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 2、右图：右前、右后
Mat frame_rrf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_rrb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 3、前图：左前、右前
Mat frame_flf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_frf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 4、后图：左后、右后
Mat frame_blb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_brb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 5、刻画上述角图尺寸，用于copyTo函数占位
Rect T(0, 0, 240, 120);
// -------------------------------------定义图像背景，用于最终显示-----------------------------------
Mat frame_ground = Mat::Mat(480, 720, CV_8UC3, (0, 0, 0));
// ------------------------------------------暂时保存角点图像----------------------------------------
Mat temp_lf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_lb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_rf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_rb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// ---------------------------------------定义时间，用于计算FPS--------------------------------------
clock_t begin_time, end_time;

// --------------------------------------------显示侧面图像-------------------------------------------
void show_side(VideoCapture path, Mat& frame, Mat& frame1, Mat& frame2, int dir) {
	path >> frame;
	switch (dir) {
	case(L):
		if (frame.empty()) {
			cout << "视频读取完成" << endl;
			break;
		}
		frame(LEFT).copyTo(frame_ground(G_LEFT));
		frame(LEFT_LF).copyTo(frame1(T));
		frame(LEFT_LB).copyTo(frame2(T));
		break;
	case(R):
		if (frame.empty()) {
			cout << "视频读取完成" << endl;
			break;
		}
		frame(RIGHT).copyTo(frame_ground(G_RIGHT));
		frame(RIGHT_RF).copyTo(frame1);
		frame(RIGHT_RB).copyTo(frame2);
		break;
	case(F):
		if (frame.empty()) {
			cout << "视频读取完成" << endl;
			break;
		}
		frame(FRONT).copyTo(frame_ground(G_FRONT));
		frame(FRONT_LF).copyTo(frame1);
		frame(FRONT_RF).copyTo(frame2);
		break;
	case(B):
		if (frame.empty()) {
			cout << "视频读取完成" << endl;
			break;
		}
		frame(BACK).copyTo(frame_ground(G_BACK));
		frame(BACK_LB).copyTo(frame1);
		frame(BACK_RB).copyTo(frame2);
		break;
	}
}
// --------------------------------------------显示角图像--------------------------------------------
void show_corner(Mat frame1, Mat frame2, Mat& frame, int dir) {
	switch (dir) {
	case(L):
		/*****************************************************************************************
		****************************************加权融合******************************************
		**********************addWeighted(frame1, 0.5, frame2, 0.5, 0, frame)*********************
		********************************根据距离计算权重进行融合**********************************
		*****************************************************************************************/
		for (int i=0; i < frame1.rows; i++) {
			for (int j=0; j < frame.cols; j++) {
				float mask = pow((120. - i), 2) / (pow((120. - i), 2) + pow((240. - j), 2));
				frame.at<Vec3b>(i, j) = frame2.at<Vec3b>(i, j)*mask + frame1.at<Vec3b>(i, j) * (1-mask);
			}
		}
		frame.copyTo(frame_ground(G_LF));
		break;
	case(R):

		for (int i = 0; i < frame1.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				float mask = pow((120 - i), 2) / (pow((120 - i), 2) + pow((j - 0), 2));
				frame.at<Vec3b>(i, j) = frame2.at<Vec3b>(i, j) * mask + frame1.at<Vec3b>(i, j) * (1 - mask);
			}
		}
		frame.copyTo(frame_ground(G_RF));
		break;
	case(F):
		for (int i = 0; i < frame1.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				float mask = pow((i - 0), 2) / (pow((i - 0), 2) + pow((240. - j), 2));
				frame.at<Vec3b>(i, j) = frame2.at<Vec3b>(i, j) * mask + frame1.at<Vec3b>(i, j) * (1 - mask);
			}
		}
		frame.copyTo(frame_ground(G_LB));
		break;
	case(B):
		for (int i = 0; i < frame1.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				float mask = pow((i - 0), 2) / (pow((i - 0), 2) + pow((j - 0), 2));
				frame.at<Vec3b>(i, j) = frame2.at<Vec3b>(i, j) * mask + frame1.at<Vec3b>(i, j) * (1 - mask);
			}
		}
		frame.copyTo(frame_ground(G_RB));
		break;
	}
}

// ------------------------------------------------主函数--------------------------------------------
int main() {
	//初始化车辆
	Mat car = imread("images/car.png");
	resize(car, car, Size(240, 240));
	car.copyTo(frame_ground(G_CAR));
	//死循环，开启显示
	while (true) {
		begin_time = clock();
		// 1、侧边图像拼接
		thread side_left(show_side, capture_left, ref(frame_left), ref(frame_llf), ref(frame_llb), L);
		thread side_right(show_side, capture_right, ref(frame_right), ref(frame_rrf), ref(frame_rrb), R);
		thread side_front(show_side, capture_front, ref(frame_front), ref(frame_flf), ref(frame_frf), F);
		thread side_back(show_side, capture_back, ref(frame_back), ref(frame_blb), ref(frame_brb), B);
		side_left.join();
		side_right.join();
		side_front.join();
		side_back.join();
		// 2、角图像融合
		thread  corner_left_front(show_corner, frame_llf, frame_flf, ref(temp_lf), L);
		thread  corner_right_front(show_corner, frame_rrf, frame_frf, ref(temp_lb), R);
		thread  corner_left_back(show_corner, frame_llb, frame_blb, ref(temp_rf), F);
		thread  corner_right_back(show_corner, frame_rrb, frame_brb, ref(temp_rb), B);
		corner_left_front.join();
		corner_right_front.join();
		corner_left_back.join();
		corner_right_back.join();
		// 图像显示，FPS打印。
		namedWindow("test", WINDOW_NORMAL);
		imshow("test", frame_ground);
		end_time = clock();
		cout << "fps = " << 1000 / (end_time - begin_time) << endl;
		waitKey(1);
	}
	return 0;
}