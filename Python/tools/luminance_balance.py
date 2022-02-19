"""
亮度平衡
"""
import cv2
import numpy as np


def luminance_balance(images):
    [img1, img2] = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in images]
    hf, sf, vf = cv2.split(img1)
    hb, sb, vb = cv2.split(img2)
    V_f = np.mean(vf)
    V_b = np.mean(vb)
    V_mean = (V_f + V_b) / 2
    vf = cv2.add(vf, (V_mean - V_f))
    vb = cv2.add(vb, (V_mean - V_b))
    img1 = cv2.merge([hf, sf, vf])
    img2 = cv2.merge([hb, sb, vb])
    images = [img1, img2]
    images = [cv2.cvtColor(image, cv2.COLOR_HSV2BGR) for image in images]
    return images