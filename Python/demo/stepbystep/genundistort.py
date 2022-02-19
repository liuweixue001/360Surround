import cv2
import yaml
import numpy as np
import time

if __name__ == "__main__":
    with open("../pic/param/DF-4105H2.yaml", "r") as f:
        d = yaml.load(f.read(), Loader=yaml.FullLoader)
        mat_inter = np.array(d["mtx"])
        coff_dis = np.array(d["dist"])
    img = cv2.imread("../pic/right/img00023.jpg")
    h, w = img.shape[:2]
    t1 = time.time()
    # new, _ = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w, h), 0, (w, h))
    # print(new)
    # dst = cv2.undistort(img, mat_inter, coff_dis, new)
    new_matrix = mat_inter
    new_matrix[0, 0] *= 0.9999
    new_matrix[1, 1] *= 0.9999
    a, b = cv2.initUndistortRectifyMap(mat_inter, coff_dis, np.eye(3),
                                     new_matrix, (w, h), cv2.CV_16SC2);
    dst = cv2.remap(img, a, b, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT);
    print(1/(time.time()-t1))
    # cv2.imwrite("left.jpg", dst)
    cv2.imshow('origin', img)
    cv2.imshow('new', dst)
    cv2.waitKey(0)
