/*#include "main.h"

//定义front图像像素点和世界坐标系坐标
Point2f pixel_f1(351, 357);
Point2f pixel_f2(236, 403);
Point2f pixel_f3(713, 325);
Point2f pixel_f4(917, 350);
Point2f world_f1(0, 0);
Point2f world_f2(0, 120);
Point2f world_f3(600, 0);
Point2f world_f4(600, 120);
//定义front点集向量
vector<Point2f> pixel_f, world_f;
//定义left图像像素点和世界坐标系坐标
Point2f pixel_l1(651, 281);
Point2f pixel_l2(705, 305);
Point2f pixel_l3(188, 293);
Point2f pixel_l4(39, 323);
Point2f world_l1(0, 0);
Point2f world_l2(80, 0);
Point2f world_l3(0, 1000);
Point2f world_l4(80, 1000);
//定义left点集向量
vector<Point2f> pixel_l, world_l;
//定义right图像像素点和世界坐标系坐标
Point2f pixel_r1(651, 281);
Point2f pixel_r2(705, 305);
Point2f pixel_r3(188, 293);
Point2f pixel_r4(39, 323);
Point2f world_r1(0, 0);
Point2f world_r2(80, 0);
Point2f world_r3(0, 1000);
Point2f world_r4(80, 1000);
//定义right点集向量
vector<Point2f> pixel_r, world_r;
//定义back图像像素点和世界坐标系坐标
Point2f pixel_b1(351, 357);
Point2f pixel_b2(236, 403);
Point2f pixel_b3(713, 325);
Point2f pixel_b4(917, 350);
Point2f world_b1(0, 0);
Point2f world_b2(0, 120);
Point2f world_b3(600, 0);
Point2f world_b4(600, 120);
//定义right点集向量
vector<Point2f> pixel_b, world_b;


int find_hor() {
	pixel_f.push_back(pixel_f1);
	pixel_f.push_back(pixel_f2);
	pixel_f.push_back(pixel_f3);
	pixel_f.push_back(pixel_f4);
	world_f.push_back(world_f1);
	world_f.push_back(world_f2);
	world_f.push_back(world_f3);
	world_f.push_back(world_f4);
	pixel_l.push_back(pixel_l1);
	pixel_l.push_back(pixel_l2);
	pixel_l.push_back(pixel_l3);
	pixel_l.push_back(pixel_l4);
	world_l.push_back(world_l1);
	world_l.push_back(world_l2);
	world_l.push_back(world_l3);
	world_l.push_back(world_l4);
	pixel_r.push_back(pixel_r1);
	pixel_r.push_back(pixel_r2);
	pixel_r.push_back(pixel_r3);
	pixel_r.push_back(pixel_r4);
	world_r.push_back(world_r1);
	world_r.push_back(world_r2);
	world_r.push_back(world_r3);
	world_r.push_back(world_r4);
	pixel_b.push_back(pixel_b1);
	pixel_b.push_back(pixel_b2);
	pixel_b.push_back(pixel_b3);
	pixel_b.push_back(pixel_b4);
	world_b.push_back(world_b1);
	world_b.push_back(world_b2);
	world_b.push_back(world_b3);
	world_b.push_back(world_b4);
	Mat homogr_l, homogr_r, homogr_f, homogr_b;

	homogr_l = findHomography(pixel_l, world_l);
	homogr_r = findHomography(pixel_r, world_r);
	homogr_f = findHomography(pixel_f, world_f);
	homogr_b = findHomography(pixel_b, world_b);
	cout << homogr_l << endl;
	cout << homogr_r << endl;
	cout << homogr_f << endl;
	cout << homogr_b << endl;
	return 0;
}*/