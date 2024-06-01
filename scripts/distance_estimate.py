#! /usr/bin/env python

import cv2
import numpy as np


# 该类主要进行双目的视差图生成、三维点云生成、点云距离判断等三维重建操作
class Depth3D:
    # Rotation Matrices 矫正后的旋转矩阵
    __R1: np.ndarray = None
    __R2: np.ndarray = None

    # Translation Matrices 矫正后的平移矩阵
    __T1: np.ndarray = None
    __T2: np.ndarray = None

    # Projection Matrices 投影矩阵
    __P1: np.ndarray = None
    __P2: np.ndarray = None

    # Q matrix
    __Q: np.ndarray = None

    # 左相机矫正映射
    left_map1: np.ndarray = None
    left_map2: np.ndarray = None

    # 右相机矫正映射
    right_map1: np.ndarray = None
    right_map2: np.ndarray = None

    # 该函数用于生成双目视差图像
    def get_stereo_disparity(self, left_frame: np.ndarray, right_frame: np.ndarray):
        # 将图像预处理为灰度图
        left_gray = None
        right_gray = None
        cv2.cvtColor(left_frame, left_gray, cv2.COLOR_BGR2GRAY)
        cv2.cvtColor(right_frame, right_gray, cv2.COLOR_BGR2GRAY)

        # 建立立体视觉匹配对象
        stereo = cv2.StereoBM_create()

        '''
            暂未开放
        '''

        '''
        # 对左、右图像进行矫正
        left_rectified = cv2.remap(left_gray, left_remap_x, left_remap_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        right_rectified = cv2.remap(right_gray, right_remap_x, right_remap_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        '''

        # 设置视差计算参数
        '''
            此处在未来应当根据实际情况进行调整，对代码进行解耦合
        '''

        # numDisparities应该是16的倍数，从16开始，这里设置为16*9
        stereo.setNumDisparities(16 * 9)

        # 设置blockSize的大小，从5开始，这里设置为21x21
        stereo.setBlockSize(21)

        # preFilterType设置为Sobel滤波
        stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)

        # 设置预处理滤波器窗口大小，从5开始，这里取9
        stereo.setPreFilterSize(9)

        # 设置预处理滤波器的容量，取值范围[1, 63]，这里取50
        stereo.setPreFilterCap(50)

        # 设置纹理阈值的大小，通常取值[0, 500]，这里取100
        stereo.setTextureThreshold(100)

        # 设置唯一性比率，取值范围[5, 15]，这里取10
        stereo.setUniquenessRatio(10)

        # 设置杂波范围，取值范围[0, 100]，这里取10
        stereo.setSpeckleRange(10)

        # 设置杂波窗口大小，取值范围[0, 100]，这里取10
        stereo.setSpeckleWindowSize(10)

        # 设置视察计算的最大差异值，通常取值[0, 100]，这里取50
        stereo.setDisp12MaxDiff(50)

        # 设置最小视差值，通常取值[0, 100]，这里取10
        stereo.setMinDisparity(10)

        # 计算视差图
        # 注意：这里的disparity是一个CV_16S的单通道图像，因此需要转换为CV_32F，然后缩放至原来的1/16
        disparity = stereo.compute(left_gray, right_gray)

        # 类型转换
        disparity = disparity.astype(np.float32) / 16.0

        # 缩放并获取视差图
        # 10即为mimDisparity的值，16 * 9即为numDisparities的值
        disparity = (disparity / 16 - 10) / (16 * 9)

        return disparity

    # 该函数用于从视差图中生成三维点云
    def estimate_3d_coordinates(self, disparity: np.ndarray):
        # 生成点云
        point_cloud_matrix = cv2.reprojectImageTo3D(disparity, self.__Q, False, -1)
        return point_cloud_matrix

    # 该函数用于比较两个不同方向映射到三维空间中的点的欧几里得距离是否小于阈值
    def is_point_close_enough(self, point1: np.ndarray, point2: np.ndarray, threshold: float):
        # 计算两点之间的欧几里得距离
        distance_3d = np.linalg.norm(point1 - point2)
        # 或者使用下面的方式
        # distance_3d = cv2.norm(point1, point2, cv2.NORM_L2)
        return distance_3d <= threshold

    # 构造函数，需要传入双目相机各个相机的内参矩阵、畸变系数、旋转矩阵、平移矩阵、图像尺寸（单画面尺寸）等参数
    def __init__(self, camera_matrix1: np.ndarray, camera_matrix2: np.ndarray, dist_coeffs1: np.ndarray,
                 dist_coeffs2: np.ndarray, R: np.ndarray, T: np.ndarray, image_size: tuple):
        # 执行矫正操作，获取矫正后的旋转矩阵、投影矩阵、视差矩阵等
        self.__R1, self.__R2, self.__P1, self.__P2, self.__Q, roi1, roi2 = cv2.stereoRectify(camera_matrix1,
                                                                                             dist_coeffs1,
                                                                                             camera_matrix2,
                                                                                             dist_coeffs2, image_size,
                                                                                             R,
                                                                                             T,
                                                                                             flags=cv2.CALIB_ZERO_DISPARITY)
        # 获取矫正后的平移矩阵
        self.__T1 = self.__P1[:, 3]
        self.__T2 = self.__P2[:, 3]

        # 生成矫正映射表
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, self.__R1, self.__P1,
                                                                     image_size, cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, self.__R2,
                                                                       self.__P2, image_size, cv2.CV_32FC1)

    # 析构函数
    def __del__(self):
        pass


# 该类主要用于计算物体距离双目相机的距离（三角相似原理）
class stereo_distance_calc:
    # 两个目标在不同相机画面下的像素坐标差
    __delta_x_pixel: float = 0.0

    # 双目相机的基线距离
    __two_cameras_dist: float = 0.0

    # 相机的焦距
    __focal_length: float = 0.0

    # 该函数用于计算距离
    def calc_distance(self):
        # 利用相似原理计算距离
        # 注意：这里的delta_x_pixel是两个目标在不同相机画面下的像素坐标差，同时注意单位统一（默认为：mm）
        distance = (self.__focal_length * self.__two_cameras_dist) / self.__delta_x_pixel
        return distance

    # 构造函数，需要传入两个目标在不同相机画面下的像素坐标、双目相机的基线距离、相机的焦距等参数
    def __init__(self, left_object_point: tuple, right_object_point: tuple, distance_between_cameras: float,
                 focal_length: float):
        # 计算两个目标在不同相机画面下的像素坐标差
        self.__delta_x_pixel = abs(left_object_point[0] - right_object_point[0])
        # 设置双目相机的基线距离
        self.__two_cameras_dist = distance_between_cameras
        # 设置相机的焦距
        self.__focal_length = focal_length

    # 析构函数
    def __del__(self):
        pass
