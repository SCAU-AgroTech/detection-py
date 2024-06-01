#! /usr/bin/env python

import cv2
import numpy as np

# 该Dictionary用于指示保存的格式
save_format = {'YAML': '.yaml', 'XML': '.xml', 'JSON': '.json'}


# 该类主要用于保存OpenCV标定的数据文件（以.yaml或.xml格式进行保存）
class SaveCalibrationData:
    # 保存标定数据的目录的字段
    __directory: str = None

    # 保存标定数据格式的字段
    __format: str = save_format['YAML']

    # 该函数用于执行保存标定数据的操作
    def save_data(self, file_name: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        # 调用cv2.FileStorage函数创建一个文件存储对象，并根据选择来保存为.yaml或.xml格式
        fs = cv2.FileStorage(self.__directory + file_name + self.__format, cv2.FILE_STORAGE_WRITE)
        # 执行写入
        fs.write('camera_matrix', camera_matrix)
        fs.write('dist_coeffs', dist_coeffs)
        # 释放文件存储对象
        fs.release()
        # 输出保存成功的信息
        print('已成功将标定文件保存至：' + self.__directory + file_name + self.__format)

        # 构造函数
        def __init__(self, directory: str, format: str = 'YAML'):
            self.__directory = directory
            self.__format = save_format[format]

    # 析构函数
    def __del__(self):
        pass
