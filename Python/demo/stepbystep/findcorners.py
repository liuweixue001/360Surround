import cv2
img = cv2.imread("../left.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray_img.shape[:2]
cv2.imshow("hh", gray_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
ret, cp_img = cv2.findChessboardCorners(gray_img, (3, 3), None)
if ret:
    for i in range(0, len(cp_img)):
        if i in [0, 2, 6, 8]:
            print(cp_img[i])