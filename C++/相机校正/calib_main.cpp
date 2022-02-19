#define _CRT_SECURE_NO_WARNINGS
#include <opencv2\opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	// ---------------------------------------------提取图片-----------------------------------------
	cout << "开始提取角点………………" << endl;
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
		// ------------------------------------------提取角点-----------------------------------------
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
	cout << "角点提取完成！\n";
	// ---------------------------------------------相机标定-----------------------------------------
	cout << "开始定标………………" << endl;
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
	cout << "定标完成！\n";
	cout << intrinsic_matrix << endl;
	cout << distortion_coeffs << endl;
	// ---------------------------------------------保存参数-----------------------------------------
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
	// ---------------------------------------------测试误差-----------------------------------------
	cout << "开始评价定标结果………………" << endl;
	double total_err = 0.0;                   
	double err = 0.0;                    
	vector<Point2f>  image_points2;          
	cout << "每幅图像的定标误差：" << endl;
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
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	cout << "评价完成！" << endl;
	// ---------------------------------------------保存结果-----------------------------------------
	cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout << "相机内参数矩阵：" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数：\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rotation_vectors[i] << endl;
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;
	fout.close();
	// ---------------------------------------保存去畸变图像----------------------------------------
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
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
	cout << "保存结束" << endl;
	// ----------------------------------------测试一张图片-----------------------------------------
	if (1)
	{
		cout << "开始校正测试" << endl;
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
		cout << "保存结束" << endl;
	}
	return 0;
}