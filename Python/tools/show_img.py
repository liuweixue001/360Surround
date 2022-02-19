"""
显示原始图像、变换图像、拼接图像
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools.color_balance import color_balance


def show_image(base_img, change_img, changed_img, _img, dst_target, cb=False):
    fig = plt.figure(tight_layout=True, figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("base_img")
    ax.imshow(base_img)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("change_img")
    ax.imshow(change_img)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("changed_img")
    ax.imshow(changed_img)
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("dst_target")
    ax.imshow(_img)
    ax = fig.add_subplot(gs[2, :])
    ax.set_title("image fuse")
    if cb:
        dst_target = color_balance(dst_target)
        ax.imshow(dst_target)
    else:
        ax.imshow(dst_target)
    plt.show()