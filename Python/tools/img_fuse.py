"""
图像特征融合
"""
import numpy as np
import cv2

# 图像特征融合
def feature_fuse(base_img, changed_img):
    # 灰度图+二值化
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    _, base_img_mask = cv2.threshold(base_img_gray, 0, 255, cv2.THRESH_BINARY)
    changed_img_gray = cv2.cvtColor(changed_img, cv2.COLOR_BGR2GRAY)
    _, changed_img_mask = cv2.threshold(changed_img_gray, 0, 255, cv2.THRESH_BINARY)
    mask_all = cv2.bitwise_and(base_img_mask, changed_img_mask)
    # 计算加权mask
    lineA = np.array([[138, 77], [319, 391]])
    lineB = np.array([[630, 35], [427, 415]])
    maskP = new_get_blend_mask(base_img_mask, changed_img_mask, lineA, lineB)
    # 取出各图重叠区域部分
    base_img_cut = cv2.bitwise_and(base_img, base_img, mask=mask_all)
    changed_img_cut = cv2.bitwise_and(changed_img, changed_img, mask=mask_all)
    # 计算重叠区域的融合图像
    maskP = np.expand_dims(maskP, 2).repeat(3, axis=2)
    dst = (base_img_cut * (1 - maskP) + maskP * changed_img_cut).astype(np.uint8)
    # 抠掉图像重叠区域，并将上述图片与融合图片叠加
    base_img_ = cv2.bitwise_xor(base_img, base_img_cut)
    changed_img_ = cv2.bitwise_xor(changed_img, changed_img_cut)
    dst = base_img_ + changed_img_ + dst
    return dst

# 两点计算直线
def getline(first_x, first_y, second_x, second_y):
    A = second_y - first_y
    B = first_x - second_x
    C = second_x*first_y - first_x*second_y
    return A, B, C

# 逐点计算点到直线距离
def get_blend_mask(maskA, maskB, lineA, lineB):
    maskP = np.zeros_like(maskA).astype(np.float32)
    overlap = cv2.bitwise_and(maskA, maskB)  # 重叠区域
    indices = np.where(overlap != 0)  # 重叠区域的坐标索引

    for y, x in zip(*indices):
        distA = cv2.pointPolygonTest(np.array(lineA), (x, y), True)  # 到重叠区域边缘的距离A
        distB = cv2.pointPolygonTest(np.array(lineB), (x, y), True)  # 到重叠区域边缘的距离B
        maskP[y, x] = distA ** 2 / (distA ** 2 + distB ** 2 + 1e-6)  # 根据距离的平方比值确定该处权重
    return maskP

# 矩阵形式计算点到直线距离
def new_get_blend_mask(maskA, maskB, lineA, lineB):
    overlap = cv2.bitwise_and(maskA, maskB)  # 重叠区域
    # 找出重叠区域坐标
    search = np.argwhere(overlap > -1).reshape(420, 720, 2)
    # 定义融合线
    A1, B1, C1 = getline(lineA[0, 1], lineA[0, 0], lineA[1, 1],  lineA[1, 0])
    A2, B2, C2 = getline(lineB[0, 1], lineB[0, 0], lineB[1, 1],  lineB[1, 0])
    # 计算重叠区域点到两条直线的距离
    maskX = np.abs(A1*search[:, :, 0] + B1*search[:, :, 1] + C1)/np.sqrt(A1**2+B1**2)
    maskY = np.abs(A2*search[:, :, 0] + B2*search[:, :, 1] + C2)/np.sqrt(A2**2+B2**2)
    # 根据权重计算mask
    maskP = maskX ** 2 / (maskX ** 2 + maskY ** 2 + 1e-8)
    return maskP