import numpy as np
import cv2


img = cv2.imread("../left.jpg")
# img = cv2.imread("./pic/IR_camera_calib_img/100010.png")
h, w = img.shape[:2]
right = np.array([[255.29843, 320.88596], [244.51465, 377.58975], [158.37581, 383.47665],
                 [181.46255, 324.67996]], dtype=np.float32)
left = np.array([[514.0161,  350.66174], [592.3235, 358.3854], [619.7638,  421.51852],
                [526.5501,  410.92737]], dtype=np.float32)
# left
obj = 15*(np.array([[2, 0], [2, 2], [0, 2], [0, 0]], dtype=np.float32)+25)
# right
# obj = 15*(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)+25)
M1, mask = cv2.findHomography(left, obj)
print(left)
print(obj)



# M = cv2.getPerspectiveTransform(right, obj)
result = cv2.warpPerspective(img, M1, (720, 480))
cv2.imwrite("result_change.jpg", img)
cv2.imshow("img", img)
# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()