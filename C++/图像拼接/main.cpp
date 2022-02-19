/*
ͼ��ƴ��
1������ͼƬ��Ϊ��Ƶ
2����������ڲΡ�����ϵ����ͼ��ߴ�Ȳ���
3��������ʼ��
4����ʼ�ĸ��߳̽���ͼ��ȥ���䡢���ӱ任��ͼ��ƴ��
5�����ڸ��ӱ任���ͼ����ڽϴ���δ����ͼ���ں�
*/
#include "main.h"

using namespace std;
using namespace cv;


// ------------------------------------------��ȡ��Ƶ------------------------------------------------
VideoCapture capture_left("videos/left.avi");
VideoCapture capture_right("videos/right.avi");
VideoCapture capture_front("videos/front.avi");
VideoCapture capture_back("videos/back.avi");
// --------------------------------------����ͼƬ�ߴ缰λ��------------------------------------------
//1��ͼƬ�ڱ����еķ���λ��
Rect G_LEFT(0, 0, 80, 1000);
Rect G_RIGHT(520, 0, 80, 1000);
Rect G_FRONT(0, 0, 600, 250);
Rect G_BACK(0, 880, 600, 120);
Rect G_CAR(200, 300, 200, 400);
//2��ȥ������ͼƬ�ߴ�
Size size_left(960, 640);
Size size_right(960, 640);
Size size_front(960, 640);
Size size_back(960, 640);
//3�����ͼ�ߴ磬������ͼ�����������ͼ��ʾ��������
Size size_left_c(80, 1000);
Size size_right_c(80, 1000);
Size size_front_c(600, 250);
Size size_back_c(600, 120);
// ------------------------------------���������������Ӧ�Ծ���--------------------------------------
//�����ĸ���λ����ͷ���ڲκͻ���ϵ��
/****************************************************************************************************
**********************************Ĭ�������opencv���ͼ����вü�***********************************
*******************************�޸ĵ����ڲε�һ�����ϵ��������ʾ����********************************
****************************************************************************************************/
//left
Matx33d intrinsic_matrix_left((3.0334009006384287e2), 0., (4.8649280066241465e2), 0.,
	(3.2229678244636966e2), (3.2388095214561167e2), 0., 0., 1);
Matx33d new_intrinsic_matrix_left((3.0334009006384287e2)*0.3, 0., (4.8649280066241465e2), 0.,
	(3.2229678244636966e2)*0.3, (3.2388095214561167e2), 0., 0., 1);
Vec4d distortion_coeffs_left(-3.5510560636666778e-2, -1.9848228876245811e-2,
	2.6080053057044101e-2, -9.7183762742328750e-3);
//right
Matx33d intrinsic_matrix_right(3.0290778983957682e+02, 0., 4.5799765697290070e+02, 0.,
	3.2250139109237318e+02, 3.1001321001054703e+02, 0., 0., 1.);
Matx33d new_intrinsic_matrix_right((3.0290778983957682e+02) * 0.3, 0., 4.5799765697290070e+02, 0.,
	(3.2250139109237318e+02) * 0.3, 3.1001321001054703e+02, 0., 0., 1.);
Vec4d distortion_coeffs_right(-4.1177772399310822e-02, 4.6179881138489094e-03,
	-4.4499171471619296e-03, 8.2316738506075550e-04);
//front
Matx33d intrinsic_matrix_front(3.0245305983229298e+02, 0., 4.9664001463163459e+02, 0.,
	3.2074618594392325e+02, 3.3119980984361649e+02, 0., 0., 1.);
Matx33d new_intrinsic_matrix_front((3.0245305983229298e+02)*0.5, 0., 4.9664001463163459e+02, 0.,
	(3.2074618594392325e+02)*0.5, 3.3119980984361649e+02, 0., 0., 1.);
Vec4d distortion_coeffs_front(-4.3735601598704078e-02, 2.1692522970939803e-02,
	-2.6388839028513571e-02, 8.4123126605702321e-03);
//back
Matx33d intrinsic_matrix_back(3.0434907840374234e+02, 0., 4.8133979392511606e+02, 0.,
	3.2477726176795460e+02, 3.1646476882040702e+02, 0., 0., 1.);
Matx33d new_intrinsic_matrix_back((3.0434907840374234e+02)*0.5, 0., 4.8133979392511606e+02, 0.,
	(3.2477726176795460e+02)*0.5, 3.1646476882040702e+02, 0., 0., 1.);
Vec4d distortion_coeffs_back(-4.1568299226312187e-02, 3.1480645089822291e-03,
	-2.3982702848139551e-03, 2.3821781880039081e-05);
//��Ӧ�Ծ���,�������Ҳ�����ͼ��Ӧ�Ծ���ǰ��ͼ����ǰͼ��Ӧ�Ծ���
Matx33d homogr_l(-0.02892153580692206, -1.115889256550409, 332.3928009009713,
	0.5160539430516784, -1.161121371866279, -9.676011432218226,
	-9.839075628025882e-05, -0.004212863378548237, 0.9999999999999999);
Matx33d homogr_r(-0.02892153580692206, -1.115889256550409, 332.3928009009713,
	0.5160539430516784, -1.161121371866279, -9.676011432218226,
	-9.839075628025882e-05, -0.004212863378548237, 0.9999999999999999);
Matx33d homogr_f(-0.2942811696997599, -0.7357029242494005, 365.9386345216518,
	-0.06631432906111424, -0.7501808475038539, 291.090892059327,
	-0.0002892812640274082, -0.002867860333868753, 1);
Matx33d homogr_b(-0.2942811696997599, -0.7357029242494005, 365.9386345216518,
	-0.06631432906111424, -0.7501808475038539, 291.090892059327,
	-0.0002892812640274082, -0.002867860333868753, 1);
// ----------------------------------------����ͼ�񼰳ߴ�--------------------------------------------
//ǰ�������Լ�����ͼ��
Mat frame_left, frame_right, frame_front, frame_back;
//ͼ�񱳾�������������ʾ
Mat frame_ground = Mat::Mat(1000, 600, CV_8UC3, (0, 0, 0));

// --------------------------------------����ʱ�����ڼ���Fps-----------------------------------------
clock_t begin_time, end_time;

// -----------------------------------------ͼ������---------------------------------------------
void video_show(VideoCapture path, Mat &frame, int dir) {
	path>>frame;
	switch (dir) {
	case(L):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		undistortion(frame, intrinsic_matrix_left, distortion_coeffs_left, size_left, new_intrinsic_matrix_left);
		gen_bird(frame, size_left_c, homogr_l);
		frame.copyTo(frame_ground(G_LEFT));
		break;
	case(R):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		undistortion(frame, intrinsic_matrix_right, distortion_coeffs_right, size_right, new_intrinsic_matrix_right);
		gen_bird(frame, size_right_c, homogr_r);
		frame.copyTo(frame_ground(G_RIGHT));
		break;
	case(F):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		undistortion(frame, intrinsic_matrix_front, distortion_coeffs_front, size_front, new_intrinsic_matrix_front);
		gen_bird(frame, size_front_c, homogr_f);
		frame.copyTo(frame_ground(G_FRONT));
		break;
	case(B):
		if (frame.empty()) {
			cout << "��Ƶ��ȡ���" << endl;
			break;
		}
		undistortion(frame, intrinsic_matrix_back, distortion_coeffs_back, size_back, new_intrinsic_matrix_back);
		gen_bird(frame, size_back_c, homogr_b);
		frame.copyTo(frame_ground(G_BACK));
		break;
	}
}

// ------------------------------------------------������-------------------------------------------
int main() {
	//��ʼ������
	Mat car = imread("images/car.png");
	resize(car, car, Size(200, 400));
	car.copyTo(frame_ground(G_CAR));
	//��ѭ����������ʾ
	while (true) {
		begin_time = clock();
		/********************************************************************************************
		****************************************�������߳�*******************************************
		******************ʹ�����ݲ�ͬ���ڴ治������8��CPU�����ĸ��̣߳��ٶȿ��ı�*******************
		*************************����̵߳���ͬһ����������������Ӱ���ٶ�****************************
		********************************************************************************************/
		thread test_left(video_show, capture_left, ref(frame_left), L);
		thread test_right(video_show, capture_right, ref(frame_right), R);
		thread test_front(video_show, capture_front, ref(frame_front), F);
		thread test_back(video_show, capture_back, ref(frame_back), B);
		test_left.join();
		test_right.join();
		test_front.join();
		test_back.join();
		//ͼ����ʾ��FPS��ӡ��
		namedWindow("test", WINDOW_NORMAL);
		imshow("test", frame_ground);
		end_time = clock();
		cout <<"fps = " <<1000/(end_time - begin_time) << endl;
		waitKey(1);
	}
	return 0;
}