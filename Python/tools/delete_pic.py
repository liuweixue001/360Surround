import os
import cv2

# ------------------------------------------删除检测不到棋盘格的图片---------------------------------------------
def delete_pic(inter_corner_shape, img_dir):
    imgs_dir = os.listdir(img_dir)
    for fname in imgs_dir:
        img = cv2.imread(os.path.join(img_dir+fname))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, cp_img = cv2.findChessboardCorners(gray_img,
            (inter_corner_shape[0], inter_corner_shape[1]), None)
        if not ret:
            os.remove(fname)

def main():
    inter_corner_shape = (9, 6)
    img_dir = "./data/4"
    img_type = "jpg"
    delete_pic(inter_corner_shape, img_dir)


if __name__ == '__main__':
    main()

