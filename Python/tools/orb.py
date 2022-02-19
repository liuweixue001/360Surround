"""
特征匹配
透视变换
返回值：
    _img--匹配图
    base_img--基准图
    change_img--变换图
    changed_img--变换后的图
"""
import cv2
import numpy as np


def orb(base_img, change_img, GOOD_POINTS_LIMITED=0.99):
    # 检测特征点
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(base_img, None)
    kp2, des2 = orb.detectAndCompute(change_img, None)
    # 特征点匹配
    bf = cv2.BFMatcher.create()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # 特征点筛选
    goodPoints = []
    for i in range(len(matches) - 1):
        if matches[i].distance < GOOD_POINTS_LIMITED * matches[i + 1].distance:
            goodPoints.append(matches[i])
    # goodPoints = matches[:20] if len(matches) > 20   else matches[:]
    # print(goodPoints)
    _img = cv2.drawMatches(base_img, kp1, change_img, kp2, goodPoints, flags=2, outImg=None)
    # 根据特征点坐标计算单应性矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RHO)

    # 获取原图像的高和宽
    h1, w1, p1 = change_img.shape
    h2, w2, p2 = base_img.shape
    h = np.maximum(h1, h2)
    w = np.maximum(w1, w2)
    _movedis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))
    # 拼接图像透视变化，两幅图一样大，直接搞
    # imageTransform = cv2.warpPerspective(img_right, M, (w1 + w2 - _movedis, h))
    changed_img = cv2.warpPerspective(change_img, M, (w1, h1))
    M1 = np.float32([[1, 0, 0], [0, 1, 0]])
    # dst1 = cv2.warpAffine(img_left, M1, (w1 + w2 - _movedis, h))
    # dst1 = cv2.warpAffine(img_left, M1, (w1, h))
    return _img, base_img, changed_img