#! /usr/bin/env python



import cv2
import numpy as np


# 该类提供分割双目相机的合并图像的一些API
class ImageSplitter:
    # 双目相机的原始画面
    __frame: np.ndarray = None

    # 双目相机的左画面
    __left_frame: np.ndarray = None

    # 双目相机的右画面
    __right_frame: np.ndarray = None

    # 指示是否已经分割过
    __is_split: bool = False

    # 该函数用于执行分割操作
    def split(self):
        # 如果已经分割过，则直接结束
        if self.__is_split:
            return

        # 若没有可用于分割的图像，则直接抛出异常
        if self.__frame is None:
            print('图像分割过程：无可用于分割的图像')
            raise Exception('图像分割过程：无可用于分割的图像')

        ''' 分割左右画面
            由于双目相机的原始画面是左右相连的，因此只需要将原始画面的一半分割出来即可
            第一个:代表行，第二个:代表列，请注意上下限的取值
            shape[0]代表行数，shape[1]代表列数
        '''
        self.__left_frame = self.__frame[:, :self.__frame.shape[1] // 2]
        self.__right_frame = self.__frame[:, self.__frame.shape[1] // 2:]

        # 标记已经分割过
        self.__is_split = True

    # 该函数用于获取左画面
    def get_left_frame(self):
        # 若为None，则直接抛出异常
        if self.__left_frame is None:
            raise Exception('图像分割过程：无法获得左画面')
        else:
            return self.__left_frame

    # 该函数用于获取右画面
    def get_right_frame(self):
        # 若为None，则直接抛出异常
        if self.__right_frame is None:
            raise Exception('图像分割过程：无法获得右画面')
        else:
            return self.__right_frame

    # 构造函数
    def __init__(self, original_frame):
        # 设定初始状态
        self.__frame = original_frame
        self.__is_split = False
        self.split()

    # 析构函数，大多数情况下不需要手动执行
    def __del__(self):
        pass


'''
    游离的函数
'''


# 该函数用于提取特定区域的图像，需要传入一个区域左上角的(x, y)坐标和区域的宽度和高度
def extract_image(original_image: np.ndarray, x1: int, y1: int, width: int, height: int):
    roi = original_image[y1:y1 + height, x1:x1 + width]
    return roi


# 该函数用于将图像调整至一个特定的尺寸，不足的位置使用特定的颜色进行填充
def resize_image_to_new(image: np.ndarray, new_shape_size: tuple, color: tuple, is_auto: bool = True,
                        is_scale_up: bool = False, stride: int = 32):
    # 先获取图像的宽度和高度
    width = image.shape[1]
    height = image.shape[0]

    # 获取新的宽度和高度
    new_shape_size_height = new_shape_size[0]
    new_shape_size_width = new_shape_size[1]

    # 计算缩放比例
    scl = min(new_shape_size_height / height, new_shape_size_width / width)

    # 当不允许放大时，只允许缩小，即<=1.000
    if not is_scale_up:
        scl = min(1.000, scl)

    # 计算新的宽度和高度
    new_unpad_w = round(width * scl)
    new_unpad_h = round(height * scl)

    # 计算变化量
    dw = new_shape_size[1] - new_unpad_w
    dh = new_shape_size[0] - new_unpad_h

    # 如果是自动的，则需要对变化量进行修正
    if is_auto:
        dw %= stride
        dh %= stride

    # 整除2
    dw /= 2
    dh /= 2

    # 进行缩放
    dst = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    # 修正误差
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # 对不足处进行填充
    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return dst
