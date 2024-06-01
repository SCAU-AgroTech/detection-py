#! /usr/bin/env python

import cv2
import numpy as np


# 定义读取标定数据错误的异常类
class ReadCalibrationDataError(Exception):
    pass


# 该类用于读取来自先前通过OpenCV API保存的标定数据
class ReadCalibrationData:
    # 内参矩阵的字段
    camera_matrix: np.ndarray = None

    # 畸变系数的字段
    dist_coeffs: np.ndarray = None

    # 该函数用于从标定数据中获取CameraMatrix内参矩阵
    def get_camera_matrix(self):
        return self.camera_matrix

    # 该函数用于从标定数据中获取DistCoeffs畸变系数
    def get_dist_coeffs(self):
        return self.dist_coeffs

    # 构造函数
    def __init__(self, data_path: str):
        # 构造读取标定数据的对象
        fs = cv2.FileStorage(data_path, cv2.FILE_STORAGE_READ)

        # 判空
        if fs.isOpened() is False:
            print('读取标定数据：' + data_path + '失败！')
            raise ReadCalibrationDataError('读取标定数据：' + data_path + '失败！')

        # 从标定数据中获取CameraMatrix内参矩阵
        self.camera_matrix = fs.getNode('camera_matrix').mat()
        # 从标定数据中获取DistCoeffs畸变系数
        self.dist_coeffs = fs.getNode('dist_coeffs').mat()
        # 释放文件存储对象
        fs.release()
        # 打印成功信息
        print('读取标定数据：' + data_path + '成功！')

    # 析构函数
    def __del__(self):
        pass
