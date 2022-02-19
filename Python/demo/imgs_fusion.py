import cv2
import numpy as np
from tools import *


# ----------------------------------------------------------图像融合类----------------------------------------------------
class Imgs_Fusion:
    """
    输入图像
    img1 = "./img1.jpg"
    img2 = ",.img2.jpg"
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pointsdict = {}
        self.imgsdict = {}
        self.findcorners()
        self.obj_left = 15 * (np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32) + 25)
        self.obj_right = 15 * (np.array([[2, 0], [2, 2], [0, 0], [0, 2]], dtype=np.float32) + 25)

    # 找角点函数
    def findcorners(self):
        for i in self.kwargs:
            point_lists = []
            img = cv2.imread(self.kwargs[i])
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, cp_img = cv2.findChessboardCorners(gray_img, (3, 3), None)
            if ret:
                for j in range(0, len(cp_img)):
                    if j in [0, 2, 6, 8]:
                        point_lists.append(list(cp_img[j].ravel()))
            self.pointsdict[i] = point_lists
        return self.pointsdict

    # 生成鸟瞰图函数
    def genbird(self):
        for i in self.kwargs:
            img = cv2.imread(self.kwargs[i])
            pointslist = self.pointsdict[i]
            h, w = img.shape[:2]
            if i == "img1":
                M, mask = cv2.findHomography(np.array(pointslist), self.obj_left)
                # M = cv2.getPerspectiveTransform(pointslist, self.obj_left)
                result = cv2.warpPerspective(img, M, (w, h))
                self.imgsdict[i] = result
            elif i == "img2":
                M, mask = cv2.findHomography(np.array(pointslist), self.obj_right)
                # M = cv2.getPerspectiveTransform(pointslist, self.obj_left)
                result = cv2.warpPerspective(img, M, (w, h))
                self.imgsdict[i] = result
        return self.imgsdict

    # 图像融合函数
    def fusion(self):
        self.genbird()
        base_img, change_img = self.imgsdict
        # 棋盘格会引起匹配错乱，考虑对棋盘格区域进行裁剪
        base_img = self.imgsdict[base_img][:420, :, :]  # 基准图像
        change_img = self.imgsdict[change_img][:420, :, :]  # 拼接图像
        _img, base_img, changed_img = orb(base_img, change_img, 0.999)
        # ---------------------------------------------------按像素最大值拼接----------------------------------------------
        dst_target0 = np.maximum(base_img, changed_img)
        # -------------------------------------------------------特征融合------------------------------------------------
        dst_target1 = feature_fuse(base_img, changed_img)
        # -------------------------------------------------------画图展示------------------------------------------------
        show_image(base_img, change_img, changed_img, _img, dst_target1)


