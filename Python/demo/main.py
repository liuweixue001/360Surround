from imgs_fusion import Imgs_Fusion


def main():
    Fuse = Imgs_Fusion(img1="./left.jpg", img2="./right.jpg")
    Fuse.fusion()


if __name__ == '__main__':
    main()