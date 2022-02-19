import cv2
import os

# 分割图像
data_path = "data/origin_pic/"
imglists = os.listdir(data_path)
for i, imgs in enumerate(imglists):
    imgdir = os.path.join(data_path + imgs)
    img = cv2.imread(imgdir)
    cv2.imwrite("data/1/img_1_%04d.jpg" % i, img[:, 0:1280, :])
    cv2.imwrite("data/2/img_2_%04d.jpg" % i, img[:, 1280:2560, :])
    cv2.imwrite("data/3/img_3_%04d.jpg" % i, img[:, 2560:3840, :])
    cv2.imwrite("data/4/img_3_%04d.jpg" % i, img[:, 3840:5120, :])
cv2.destroyAllWindows()