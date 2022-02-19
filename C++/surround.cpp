#include "surround.h"
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2\opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


surround::surround(QWidget *parent)
    : QMainWindow(parent),
    ui(new Ui::surroundClass)
{
    ui->setupUi(this);
	// ѡ���ļ��У�������·��
    connect(ui->pushButton, &QPushButton::clicked, [=]() {
        filepath = QFileDialog::getExistingDirectory(this, "choose pic Directory", "path");
        });
	// ���ļ����е�ͼƬ���б궨
	connect(ui->pushButton_2, &QPushButton::clicked, [=]() {
		// ����ͼƬ��Ŀ���ǵ���Ŀ
		// demo��д��������Ui��������ѡ������Ż�
		int image_count = ui->spinBox_3->value();
		int h = ui->spinBox->value();
		int w = ui->spinBox_2->value();
		Size board_size = Size(h, w);
		// ����ǵ�����
		vector<Point2f> corners;
		vector<vector<Point2f>>corners_Seq;
		//����ͼƬ�б�
		vector<Mat>  image_Seq;
		// ��⵽�ǵ����Ƭ
		int successImageNum = 0;
		// 
		int count = 0;
		for (int i = 0; i != image_count; i++)
		{
			QString imageFileName = filepath;
			imageFileName += "/img_";
			//�ص�:�����������ַ�����ת��
			std::stringstream StrStm;
			StrStm << i + 1;
			string channel;
			StrStm >> channel;
			QString channel1 = QString::fromStdString(channel);
			imageFileName += channel1 += ".jpg";
			string path = imageFileName.toStdString();
			cv::Mat image = cv::imread(path);
			//cv::imshow("image", image);
			//cv::waitKey(10);
			cv::Mat imageGray;
			cvtColor(image, imageGray, CV_RGB2GRAY);
			//���ҽǵ�
			bool patternfound = findChessboardCorners(image, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
				CALIB_CB_FAST_CHECK);
			if (!patternfound)
			{
				qDebug() << "can not find chessboard corners!\n";
				continue;
				exit(1);
			}
			else
			{
				//�Ǿ��Ⱦ�ȷ��
				cornerSubPix(imageGray, corners, Size(3, 3), Size(-1, -1), 
					TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
				count = count + corners.size();
				successImageNum = successImageNum + 1;
				corners_Seq.push_back(corners);
			}
			image_Seq.push_back(image);
		}
		//ȷ�������������
		Size square_size = Size(20, 20);
		vector<vector<Point3f>>  object_Points;
		Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));
		vector<int>  point_counts;
		for (int t = 0; t < successImageNum; t++)
		{
			vector<Point3f> tempPointSet;
			for (int i = 0; i < board_size.height; i++)
			{
				for (int j = 0; j < board_size.width; j++)
				{
					Point3f tempPoint;
					tempPoint.x = i * square_size.width;
					tempPoint.y = j * square_size.height;
					tempPoint.z = 0;
					tempPointSet.push_back(tempPoint);
				}
			}
			object_Points.push_back(tempPointSet);
		}
		for (int i = 0; i < successImageNum; i++)
		{
			point_counts.push_back(board_size.width * board_size.height);
		}
		//����ߴ硢����ξ��󡢻���ϵ��
		Size image_size = image_Seq[0].size();
		cv::Matx33d intrinsic_matrix;
		cv::Vec4d distortion_coeffs;
		std::vector<cv::Vec3d> rotation_vectors;
		std::vector<cv::Vec3d> translation_vectors;
		int flags = 0;
		flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		flags |= cv::fisheye::CALIB_CHECK_COND;
		flags |= cv::fisheye::CALIB_FIX_SKEW;
		//��������Ρ�����ϵ��
		cv::fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, 
			distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
		//�ص㣺����stringstream���ͱ���Ϊ�м������ʵ�ִ�����������QString��ת��
		stringstream intrinsic_matrix1;
		intrinsic_matrix1 << intrinsic_matrix;
		ui->label->setText(QString::fromStdString(intrinsic_matrix1.str()));
		stringstream distortion_coeffs1;
		distortion_coeffs1 << distortion_coeffs;
		ui->label_2->setText(QString::fromStdString(distortion_coeffs1.str()));
		});

}
