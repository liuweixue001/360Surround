import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from tools import *



t1 = time.time()
img_base = cv2.imread("result_base.jpg", 1)  # 基准图像
img_change = cv2.imread("result_change.jpg", 1)  # 拼接图像
# 棋盘格会引起匹配错乱，考虑对棋盘格区域进行裁剪
img_base = img_base[:420, :, :]
img_change = img_change[:420, :, :]

print("读取数据：" + str(time.time()-t1))
# 特征匹配
img3, dst1, imageTransform = orb(img_base, img_change, 0.99)
print("特征匹配：" + str(time.time()-t1))
# # 特征融合
dst_target0 = np.maximum(dst1, imageTransform)

# 取出重叠区域
dst_target1 = cv2.bitwise_and(imageTransform, dst1)
# 灰度图+二值化+膨胀+腐蚀得到mask
dst1_gray = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
ret, dst1_mask = cv2.threshold(dst1_gray, 0, 255, cv2.THRESH_BINARY)
# dst1_mask = cv2.dilate(dst1_mask, (35, 35), iterations=6)
# dst1_mask = cv2.dilate(dst1_mask, (5, 5), iterations=8)
# dst1_mask = cv2.erode(dst1_mask, (5, 5), iterations=8)
# dst1_mask = cv2.erode(dst1_mask, (35, 35), iterations=6)
# 灰度图+二值化+膨胀+腐蚀得到mask
imageTransform_gray = cv2.cvtColor(imageTransform, cv2.COLOR_BGR2GRAY)
_, imageTransform_mask = cv2.threshold(imageTransform_gray, 0, 255, cv2.THRESH_BINARY)
# imageTransform_mask = cv2.dilate(imageTransform_mask, (25, 25), iterations=6)
# imageTransform_mask = cv2.dilate(imageTransform_mask, (5, 5), iterations=8)
# imageTransform_mask = cv2.erode(imageTransform_mask, (5, 5), iterations=8)
# imageTransform_mask = cv2.erode(imageTransform_mask, (25, 25), iterations=6)
# 获取重叠区域mask
mask_all = cv2.bitwise_and(dst1_mask, imageTransform_mask)



# 设定边界线
lineA = np.array([[138, 77], [319, 391]])
lineB = np.array([[630, 35], [427, 415]])
maskP = new_get_blend_mask(dst1_mask, imageTransform_mask, lineA, lineB)
# 取出各图重叠区域部分
dst1_cut = cv2.bitwise_and(dst1, dst1, mask=mask_all)
imageTransform_cut = cv2.bitwise_and(imageTransform, imageTransform, mask=mask_all)
# 计算重叠区域的融合图像
# maskP = np.(([maskP, maskP, maskP]), 2)
maskP = np.expand_dims(maskP, 2).repeat(3, axis=2)

dst_target1 = (imageTransform_cut*(1-maskP) + maskP*dst1_cut).astype(np.uint8)
print("图像融合：" + str(time.time()-t1))
# 验证融合结果
print((1-maskP[288, 288])*imageTransform_cut[288, 288]+maskP[288, 288]*dst1_cut[288, 288])
print(dst_target1[288, 288])
# 抠掉图像重叠区域，并将上述图片与融合图片叠加
dst1_ = cv2.bitwise_xor(dst1, dst1_cut)
imageTransform_ = cv2.bitwise_xor(imageTransform, imageTransform_cut)
dst_target1 = dst1_ + imageTransform_ + dst_target1
dst_target2 = color_balance(dst_target1)
print("图像拼接：" + str(time.time()-t1))
# 画图，enjoying
cv2.imshow("dst_target1", dst_target1)
print("opencv显示图像：" + str(time.time()-t1))
cv2.waitKey(0)

# plt画图，计算时间
t1 = time.time()
origin_base = cv2.imread("../pic/right/img00023.jpg")
origin_change = cv2.imread("../pic/left/img00021.jpg")
show_image(origin_base, origin_change, img_base, img_change, dst_target1, cb=False)
print("matplotlib画图时间：" + str(time.time()-t1))