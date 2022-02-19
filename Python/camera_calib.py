import os
import numpy as np
import cv2

# ---------------------------------------------------------相机标定类---------------------------------------------------
class Calibration:
    """
    实现普通相机标定、去畸变
    鱼眼相机标定、去畸变
    坐标轴可视化
    """
    def __init__(self,
                 inter_corner_shape,
                 size_per_grid,
                 img_dir,
                 test_img=None,
                 scale=0.666,
                 write2local=None,
                 show_time=0,
                 save_dir=None):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.size_per_grid = size_per_grid
        self.img_dir = img_dir
        self.test_img = test_img
        self.show_time = show_time
        self.obj_points = []
        self.img_points = []
        self.w, self.h = inter_corner_shape[0], inter_corner_shape[1]
        self.cp_world = self.get_cp_world()
        # ----------------------------------------------------鱼眼相关-----------------------------------------------
        self.scale = scale
        self.fish_K = np.zeros((3, 3))
        self.fish_D = np.zeros((4, 1))
        self.fish_cp_world = self.get_fish_cp_world()
        self.fish_img_shape = None
        self.write2local = write2local
        self.save_dir = save_dir

    # --------------------------------------------------------普通相机标定---------------------------------------------
    def calib(self):
        images_dir = os.listdir(self.img_dir)
        for fname in images_dir:
            img_dir = os.path.join(self.img_dir + fname)
            img = cv2.imread(img_dir)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 得到是否有角点以及角点
            ret, cp_img = cv2.findChessboardCorners(gray_img, (self.w, self.h), None)
            if ret:
                # 得到优化后的角点
                cp_img = cv2.cornerSubPix(gray_img, cp_img, (11, 11), (-1, -1), self.criteria)
                self.obj_points.append(self.cp_world)
                self.img_points.append(cp_img)
                # view the corners
                cv2.drawChessboardCorners(img, (self.w, self.h), cp_img, ret)
                cv2.imshow('FoundCorners', img)
                cv2.waitKey(100)
        cv2.destroyAllWindows()
        _, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(self.obj_points,
                                                     self.img_points, gray_img.shape[::-1], None, None)
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points_repro, _ = cv2.projectPoints(self.obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
            error = cv2.norm(self.img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
            total_error += error
        print(("Average Error of Reproject: "), total_error / len(self.obj_points))
        return mat_inter, coff_dis, self.cp_world

    # --------------------------------------------------------鱼眼相机标定---------------------------------------------
    def fishcalib(self):
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                            cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        # 获取图片
        images_dir = os.listdir(self.img_dir)
        for fname in images_dir:
            img_dir = os.path.join(self.img_dir + fname)
            img = cv2.imread(img_dir)
            # 使用第一张图片得到图像尺寸
            if self.fish_img_shape == None:
                self.fish_img_shape = img.shape[:2]
            else:
                assert self.fish_img_shape == img.shape[:2], "All images must share the same size."
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 找角点
            ret, corners = cv2.findChessboardCorners(gray_img, (self.w, self.h),
                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                self.obj_points.append(self.fish_cp_world)
                fish_cp_img = cv2.cornerSubPix(gray_img, corners, (3, 3), (-1, -1), self.criteria)
                self.img_points.append(corners)
                cv2.drawChessboardCorners(img, (self.w, self.h), fish_cp_img, ret)
                cv2.imshow('FoundCorners', img)
                cv2.waitKey(100)
        # 实际图片数
        imgs = len(self.obj_points)
        # 初始化内参、畸变系数以及旋转和平移向量
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(imgs)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(imgs)]
        # 鱼眼相机校正
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            self.obj_points, self.img_points, gray_img.shape[::-1],
            self.fish_K, self.fish_D, rvecs, tvecs, calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        fish_DIM = self.fish_img_shape[::-1]
        return fish_DIM, self.fish_K, self.fish_D

    # --------------------------------------------------------鱼眼相机坐标系--------------------------------------------
    def get_cp_world(self):
        cp_int = np.zeros((self.w * self.h, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)
        cp_world = cp_int * self.size_per_grid
        return cp_world

    # --------------------------------------------------------鱼眼相机坐标--------------------------------------------
    def get_fish_cp_world(self):
        cp_int = np.zeros((1, self.w * self.h, 3), np.float32)
        cp_int[0, :, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2).astype(np.float64)
        fish_cp_world = cp_int * self.size_per_grid
        return fish_cp_world

    # --------------------------------------------------------普通相机去畸变--------------------------------------------
    def undistort(self):
        if self.test_img:
            img = cv2.imread(self.test_img)
            mat_inter, coff_dis, _ = self.calib()
            new, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (self.w, self.h), 1, (self.w, self.h))
            # x, y, w, h = roi
            dst = cv2.undistort(img, mat_inter, coff_dis, new)
            # mapx, mapy = cv2.initUndistortRectifyMap(mat_inter, coff_dis, None, mat_inter, (self.w, self.h), 5)
            # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            cv2.imshow('origin', img)
            # cv2.imshow('new', dst[y: y+h, x: x+w])
            cv2.imshow('new', dst)
            # cv2.waitKey(self.show_time)
        else:
            print("No image was loaded!")

    # --------------------------------------------------------鱼眼相机去畸变--------------------------------------------
    def fishundistort(self):
        if self.test_img:
            img = cv2.imread(self.test_img)
            DIM, fish_K, fish_D = self.fishcalib()
            cv2.destroyAllWindows()
            dim1 = img.shape[:2][::-1]
            assert dim1[0] / dim1[1] == DIM[0] / DIM[
                1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
            if dim1[0] != DIM[0]:
                img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
            NewK = fish_K.copy()
            if self.scale:
                NewK[(0, 1), (0, 1)] = self.scale * NewK[(0, 1), (0, 1)]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(fish_K, fish_D, np.eye(3), NewK, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.namedWindow("distorted_img", cv2.WINDOW_NORMAL)
            cv2.namedWindow("undistorted_img", cv2.WINDOW_NORMAL)
            cv2.imshow("distorted_img", img)
            cv2.imshow("undistorted_img", undistorted_img)
            if self.write2local:
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                _dir = self.test_img.split('/')[-1]
                dir = self.save_dir + _dir
                cv2.imwrite(dir, undistorted_img)
            else:
                cv2.waitKey(self.show_time)

    # ---------------------------------------------------------坐标轴可视化---------------------------------------------
    def axis_visual(self):
        img = cv2.imread(self.test_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
        if ok:
            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            mat_inter, coff_dis, _ = self.calib()
            _, rvec, tvec, inliers = cv2.solvePnPRansac(self.cp_world, exact_corners, mat_inter, coff_dis)
            # rotation_m, _ = cv2.Rodrigues(rvec)
            # rotation_t = np.hstack([rotation_m, tvec])
            # rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])
            axis = 0.2 * np.float32([[0, 0, -8], [8, 0, 0], [8, 5, 0], [0, 5, 0]]).reshape(-1, 3)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mat_inter, coff_dis)
            corner = tuple(corners[0].ravel())
            img = cv2.drawContours(img, [np.array([list(corners[0].ravel()), list(imgpts[1].ravel()),
                    list(imgpts[2].ravel()), list(imgpts[3].ravel())]).astype(np.int64)], -1, (175, 0, 175), -3)
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (0, 0, 255), 5)
            cv2.imshow('img', img)
        cv2.waitKey(self.show_time)

# --------------------------------------------------------获取相机参数--------------------------------------------------
def get_DIM_K_D(inter_corner_shape, size_per_grid, img_dir, scale):
    cam = Calibration(inter_corner_shape=inter_corner_shape,
                     size_per_grid=size_per_grid,
                     img_dir=img_dir,
                     test_img=None,
                     scale=scale,
                     write2local=None,
                     show_time=0,
                     save_dir=None)
    DIM, K, D = cam.fishcalib()
    return DIM, K, D

# ---------------------------------------------------------图像去畸变---------------------------------------------------
def get_fishundistort(inter_corner_shape, size_per_grid, cam_img_dir, scale, undistort_img_dir, save_dir):
    cam = Calibration(inter_corner_shape=inter_corner_shape,
                     size_per_grid=size_per_grid,
                     img_dir=cam_img_dir,
                     test_img=None,
                     scale=scale,
                     write2local=None,
                     show_time=0,
                     save_dir=save_dir)
    DIM, K, D = cam.fishcalib()
    NewK = K.copy()
    if scale:
        NewK[(0, 1), (0, 1)] = scale * K[(0, 1), (0, 1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NewK, DIM, cv2.CV_16SC2)
    _imgs_dirs = os.listdir(undistort_img_dir)
    for _img_dir in _imgs_dirs:
        save_img_dir = _img_dir
        _imgdir = os.path.join(undistort_img_dir, _img_dir)
        _img = cv2.imread(_imgdir)
        undistorted_img = cv2.remap(_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        dir = save_dir + save_img_dir
        cv2.imwrite(dir, undistorted_img)






