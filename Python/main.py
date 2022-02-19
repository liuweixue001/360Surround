from camera_calib import Calibration, get_DIM_K_D, get_fishundistort
import argparse

# --------------------------------------------------------超参数---------------------------------------------------------
def get_arguments():
    parser = argparse.ArgumentParser('Calibration')
    parser.add_argument('-inter_corner_shape', type=tuple, default=(9, 6))
    parser.add_argument('-size_per_grid', type=float, default=0.2)
    parser.add_argument('-img_dir', type=str, default="./data/imgs/")
    parser.add_argument('-test_img', type=str, default="./data/camera_data/1/img_1_0077.jpg")
    parser.add_argument('-save_dir', type=str, default="./data/save_dir/")
    parser.add_argument('-scale', type=float, default=0.666)
    parser.add_argument('-write2local', type=bool, default=True)
    parser.add_argument('-show_time', type=int, default=0)
    arguments = parser.parse_args()
    return arguments


# 图像去畸变
def main():
    args = get_arguments()
    # --------------------------------------------------demo 去畸变----------------------------------------------------
    # get_fishundistort(inter_corner_shape=args.inter_corner_shape,
    #                   size_per_grid=args.size_per_grid,
    #                   cam_img_dir=args.img_dir,
    #                   scale=args.scale,
    #                   undistort_img_dir=args.img_dir,
    #                   save_dir=args.save_dir)
    # --------------------------------------------------demo 获取相机参数-----------------------------------------------
    _, K, D = get_DIM_K_D(inter_corner_shape=args.inter_corner_shape,
                          size_per_grid=args.size_per_grid,
                          img_dir=args.img_dir,
                          scale=None)
    print(K)
    print(D)


if __name__ == '__main__':
    main()