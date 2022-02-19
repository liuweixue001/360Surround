#define _CRT_SECURE_NO_WARNINGS
#include <opencv2\opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	// ---------------------------------------------��ȡͼƬ-----------------------------------------
	cout << "��ʼ��ȡ�ǵ㡭����������" << endl;
	int image_count = 18;                   
	Size board_size = Size(9, 6);           
	vector<Point2f> corners;                 
	vector<vector<Point2f>>  corners_Seq;   
	vector<Mat>  image_Seq;
	int successImageNum = 0;				
	int count = 0;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += ".jpg";
		cv::Mat image = imread("imgs/img_2_" + imageFileName);
		// ------------------------------------------��ȡ�ǵ�-----------------------------------------
		cv::Mat imageGray;
		cvtColor(image, imageGray, CV_RGB2GRAY);
		bool patternfound = findChessboardCorners(image, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
			CALIB_CB_FAST_CHECK);
		if (!patternfound)
		{
			cout << "can not find chessboard corners!\n";
			continue;
			exit(1);
		}
		else
		{
			cornerSubPix(imageGray, corners, Size(3, 3), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			Mat imageTemp = image.clone();
			for (int j = 0; j < corners.size(); j++)
			{
				circle(imageTemp, corners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
			}
			string imageFileName = "result/";
			std::stringstream StrStm;
			StrStm << i + 1;
			StrStm >> imageFileName;
			imageFileName += "_corner.jpg";
			imwrite("corners/" + imageFileName, imageTemp);
			cout << "Frame corner#" << i + 1 << "...end" << endl;
			count = count + corners.size();
			successImageNum = successImageNum + 1;
			corners_Seq.push_back(corners);
		}
		image_Seq.push_back(image);
	}
	cout << "�ǵ���ȡ��ɣ�\n";
	// ---------------------------------------------����궨-----------------------------------------
	cout << "��ʼ���ꡭ����������" << endl;
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
	Size image_size = image_Seq[0].size();
	cv::Matx33d intrinsic_matrix;   
	cv::Vec4d distortion_coeffs;    
	std::vector<cv::Vec3d> rotation_vectors;                         
	std::vector<cv::Vec3d> translation_vectors;                      
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
	cout << "������ɣ�\n";
	cout << intrinsic_matrix << endl;
	cout << distortion_coeffs << endl;
	// ---------------------------------------------�������-----------------------------------------
	string datFileName = "result/camParam.dat";
	FILE* camParam = fopen(datFileName.c_str(), "wb");
	if (camParam == NULL) {
		std::cout << "can not create data file: " << datFileName.c_str() << " !!!" << std::endl;
		return false;
	}
	fwrite(&intrinsic_matrix, sizeof(cv::Matx33d), 1, camParam);
	fwrite(&distortion_coeffs, sizeof(cv::Vec4d), 1, camParam);
	fwrite(&image_size, sizeof(Size), 1, camParam);
	fclose(camParam);
	// ---------------------------------------------�������-----------------------------------------
	cout << "��ʼ���۶�����������������" << endl;
	double total_err = 0.0;                   
	double err = 0.0;                    
	vector<Point2f>  image_points2;          
	cout << "ÿ��ͼ��Ķ�����" << endl;
	ofstream fout("result/caliberation_result.txt");
	for (int i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	cout << "����ƽ����" << total_err / image_count << "����" << endl;
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;
	cout << "������ɣ�" << endl;
	// ---------------------------------------------������-----------------------------------------
	cout << "��ʼ���涨����������������" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */
	fout << "����ڲ�������" << endl;
	fout << intrinsic_matrix << endl;
	fout << "����ϵ����\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << rotation_vectors[i] << endl;
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotation_matrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "��ɱ���" << endl;
	fout << endl;
	fout.close();
	// ---------------------------------------����ȥ����ͼ��----------------------------------------
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "�������ͼ��" << endl;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
		string imageFileName = "result/";
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite("result/" + imageFileName, t);
	}
	cout << "�������" << endl;
	// ----------------------------------------����һ��ͼƬ-----------------------------------------
	if (1)
	{
		cout << "��ʼУ������" << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		Mat testImage = imread("imgs/img_2_10.jpg");
		if (testImage.empty()) {
			std::cout << "No pic was found" << std::endl;
			return false;
		}
		cv::Matx33d intrinsic_matrix;
		cv::Vec4d distortion_coeffs;
		Size image_size;
		string datFileName = "result/camParam.dat";
		FILE* camParam = fopen(datFileName.c_str(), "rb");
		if (camParam == NULL) {
			std::cout << "can not create data file: " << datFileName.c_str() << " !!!" << std::endl;
			return false;
		}
		fread(&intrinsic_matrix, sizeof(cv::Matx33d), 1, camParam);
		fread(&distortion_coeffs, sizeof(cv::Vec4d), 1, camParam);
		fread(&image_size, sizeof(Size), 1, camParam);
		fclose(camParam);
		Mat mapx = Mat(image_size, CV_32FC1);
		Mat mapy = Mat(image_size, CV_32FC1);
		Mat R = Mat::eye(3, 3, CV_32F);
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		Mat t = testImage.clone();
		cv::remap(testImage, t, mapx, mapy, INTER_LINEAR);
		imshow("test", t);
		waitKey(0);
		cvDestroyAllWindows();
		cout << "�������" << endl;
	}
	return 0;
}