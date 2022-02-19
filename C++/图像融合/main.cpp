/*
ͼ���ں�
5�����ڸ��ӱ任���ͼ����ڽϴ���δ����ͼ���ں�
6�����òü�ͼƬ������ƴ�ӡ��ں�
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
// -------------------------------------------���巽λö��-------------------------------------------
enum DIRECTION { L, R, F, B } dir;
// ---------------------------------------------��ȡ��Ƶ---------------------------------------------
VideoCapture capture_left("videos/left.avi");
VideoCapture capture_right("videos/right.avi");
VideoCapture capture_front("videos/front.avi");
VideoCapture capture_back("videos/back.avi");

// -------------------------------------����ͼƬ�ڱ����еķ���λ��-----------------------------------
Rect G_LF(0, 0, 240, 120);
Rect G_LB(0, 360, 240, 120);
Rect G_RF(480, 0, 240, 120);
Rect G_RB(480, 360, 240, 120);
Rect G_LEFT(0, 120, 240, 240);
Rect G_RIGHT(480, 120, 240, 240);
Rect G_FRONT(240, 0, 240, 120);
Rect G_BACK(240, 360, 240, 120);
Rect G_CAR(240, 120, 240, 240);
// --------------------------------------------����ü�����------------------------------------------
// 1������λ
Rect LEFT(0, 120, 240, 240);
Rect RIGHT(0, 120, 240, 240);
Rect FRONT(240, 0, 240, 120);
Rect BACK(240, 0, 240, 120);
// 2���Ƿ�λ
Rect LEFT_LF(0, 0, 240, 120); //��ͼ ����
Rect FRONT_LF(0, 0, 240, 120);//ǰͼ ����
Rect RIGHT_RF(0, 0, 240, 120);//��ͼ ����
Rect FRONT_RF(480, 0, 240, 120);// ǰͼ ����
Rect LEFT_LB(0, 360, 240, 120);//��ͼ ���
Rect BACK_LB(0, 0, 240, 120);//��ͼ ���
Rect RIGHT_RB(0, 360, 240, 120);//��ͼ �Һ�
Rect BACK_RB(480, 0, 240, 120);//��ͼ �Һ�

// -------------------------------------------�������ͼ��-------------------------------------------
Mat frame_left, frame_right, frame_front, frame_back;
// -------------------------------------------����ǵ�ͼƬ-------------------------------------------
// 1����ͼ����ǰ�����
Mat frame_llf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_llb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 2����ͼ����ǰ���Һ�
Mat frame_rrf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_rrb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 3��ǰͼ����ǰ����ǰ
Mat frame_flf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_frf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 4����ͼ������Һ�
Mat frame_blb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat frame_brb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// 5���̻�������ͼ�ߴ磬����copyTo����ռλ
Rect T(0, 0, 240, 120);
// -------------------------------------����ͼ�񱳾�������������ʾ-----------------------------------
Mat frame_ground = Mat::Mat(480, 720, CV_8UC3, (0, 0, 0));
// ------------------------------------------��ʱ����ǵ�ͼ��----------------------------------------
Mat temp_lf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_lb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_rf = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
Mat temp_rb = Mat::Mat(120, 240, CV_8UC3, (0, 0, 0));
// ---------------------------------------����ʱ�䣬���ڼ���FPS--------------------------------------
clock_t begin_time, end_time;

// --------------------------------------------��ʾ����ͼ��-------------------------------------------
void show_side(VideoCapture path, Mat& frame, Mat& frame1, Mat& frame2, int dir) {
	path >> frame;
	switch (dir) {
	case(L):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		frame(LEFT).copyTo(frame_ground(G_LEFT));
		frame(LEFT_LF).copyTo(frame1(T));
		frame(LEFT_LB).copyTo(frame2(T));
		break;
	case(R):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		frame(RIGHT).copyTo(frame_ground(G_RIGHT));
		frame(RIGHT_RF).copyTo(frame1);
		frame(RIGHT_RB).copyTo(frame2);
		break;
	case(F):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		frame(FRONT).copyTo(frame_ground(G_FRONT));
		frame(FRONT_LF).copyTo(frame1);
		frame(FRONT_RF).copyTo(frame2);
		break;
	case(B):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		frame(BACK).copyTo(frame_ground(G_BACK));
		frame(BACK_LB).copyTo(frame1);
		frame(BACK_RB).copyTo(frame2);
		break;
	}
}
// --------------------------------------------��ʾ��ͼ��--------------------------------------------
void show_corner(Mat frame1, Mat frame2, Mat& frame, int dir) {
	switch (dir) {
	case(L):
		/*****************************************************************************************
		****************************************��Ȩ�ں�******************************************
		**********************addWeighted(frame1, 0.5, frame2, 0.5, 0, frame)*********************
		********************************���ݾ������Ȩ�ؽ����ں�**********************************
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

// ------------------------------------------------������--------------------------------------------
int main() {
	//��ʼ������
	Mat car = imread("images/car.png");
	resize(car, car, Size(240, 240));
	car.copyTo(frame_ground(G_CAR));
	//��ѭ����������ʾ
	while (true) {
		begin_time = clock();
		// 1�����ͼ��ƴ��
		thread side_left(show_side, capture_left, ref(frame_left), ref(frame_llf), ref(frame_llb), L);
		thread side_right(show_side, capture_right, ref(frame_right), ref(frame_rrf), ref(frame_rrb), R);
		thread side_front(show_side, capture_front, ref(frame_front), ref(frame_flf), ref(frame_frf), F);
		thread side_back(show_side, capture_back, ref(frame_back), ref(frame_blb), ref(frame_brb), B);
		side_left.join();
		side_right.join();
		side_front.join();
		side_back.join();
		// 2����ͼ���ں�
		thread  corner_left_front(show_corner, frame_llf, frame_flf, ref(temp_lf), L);
		thread  corner_right_front(show_corner, frame_rrf, frame_frf, ref(temp_lb), R);
		thread  corner_left_back(show_corner, frame_llb, frame_blb, ref(temp_rf), F);
		thread  corner_right_back(show_corner, frame_rrb, frame_brb, ref(temp_rb), B);
		corner_left_front.join();
		corner_right_front.join();
		corner_left_back.join();
		corner_right_back.join();
		// ͼ����ʾ��FPS��ӡ��
		namedWindow("test", WINDOW_NORMAL);
		imshow("test", frame_ground);
		end_time = clock();
		cout << "fps = " << 1000 / (end_time - begin_time) << endl;
		waitKey(1);
	}
	return 0;
}