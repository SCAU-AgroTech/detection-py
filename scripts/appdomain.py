#! /usr/bin/env python

import sys
import os

# 将当前工作目录添加到系统路径中
abs_path = os.path.abspath('.')
sys.path.insert(0, abs_path + "/src/plumbing_pub_sub/scripts")

import cv2  # OpenCV 4.6.0作为图像处理库
import numpy as np  # NumPy 1.19.2作为数值计算库
import rospy  # ROS Python API
import rospkg  # ROS package API
import object_detection  # 目标检测模块
import distance_estimate  # 距离估计模块
import split_image  # 图像分割模块
import read_calibration_data  # 读取标定数据模块
import save_calibration_data  # 保存标定数据模块

# 新增自定义消息模块的路径至系统路径，并导入自定义消息模块
sys.path.append('/home/adam/detection_py_ws/devel/lib/python3/dist-packages/detection_py/msg')
from _VisionInfo import VisionInfo


# 主函数
def main():
    # 初始化 ROS 节点
    rospy.init_node('Stereo_Vision', anonymous=False)

    # 创建 ROS 发布者, 发布消息类型为 VisionInfo
    publisher = rospy.Publisher('vision_info', VisionInfo, queue_size=10)

    # 设置循环频率为 10Hz
    rate = rospy.Rate(1)

    # 通过rospkg获取包路径
    package_name = "detection_py"
    ros_pack = rospkg.RosPack()
    package_directory = ros_pack.get_path(package_name)

    # 模型路径与标定图像、数据路径
    models_directory = package_directory + '/models/'
    calib_images_directory = package_directory + '/calibration_images/'
    calib_data_directory = package_directory + '/calibration_data/'

    # 创建vision_info实例
    info = VisionInfo()

    # 打开摄像头
    # 请注意摄像头编号，应该使用bash脚本来辅以判断
    camera_index: int = 0
    camera = cv2.VideoCapture(camera_index)

    # 判空
    if not camera.isOpened():
        print("无法打开摄像头" + str(camera_index))
        exit(-1)

    # 设置摄像头参数
    # 长宽比例为 16:9，长1280，宽720，帧率30fps，使用MJPEG编解码器，设置自动曝光
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)  # 640 * 2
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.5)

    # 设置用于显示图像的窗口
    cv2.namedWindow('Left', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Right', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Matches', cv2.WINDOW_AUTOSIZE)

    # 设置窗口大小
    cv2.resizeWindow('Left', 640, 480)
    cv2.resizeWindow('Right', 640, 480)
    cv2.resizeWindow('Matches', 1280, 480)

    # 从模型所在目录获取onnx模型，然后输入类别名称
    model_path = models_directory + 'A.onnx'
    net = cv2.dnn.readNetFromONNX(model_path)
    cls_names = ['Peppers-rBaC']

    # 在循环中进行图像处理
    while not rospy.is_shutdown():
        # 获取图像并进行分割处理
        ret_bool, stereo_frame = camera.read()

        # 判空
        if not ret_bool:
            # print("主循环过程：未能获取图像")
            continue

        # 分割图像，获取左、右视图
        split = split_image.ImageSplitter(stereo_frame)
        left_frame = split.get_left_frame()
        right_frame = split.get_right_frame()

        # 创建左、右检测对象
        detect_left = object_detection.ObjectDetection(model_path, left_frame, cls_names, net)
        detect_right = object_detection.ObjectDetection(model_path, right_frame, cls_names, net)

        # 执行检测并获取检测是否成功
        detect_ret_left: bool = detect_left.detect()
        detect_ret_right: bool = detect_right.detect()

        # 分别获取当前图像中检测目标最大的物体对应的bounding box
        # 在理想环境下，若左、右相机都检测到了目标的存在，那么根据近大远小的原理，在两个画面中“类似且都最大的物体”很可能就是同一个物体，且该物体距离相机最近
        left_max = int(detect_left.get_max_box_index())
        right_max = int(detect_right.get_max_box_index())

        # 获取左右检测对象的bounding box列表的大小
        left_list_size: int = len(detect_left.bounding_box_list)
        right_list_size: int = len(detect_right.bounding_box_list)

        # 只有当两个列表的长度都大于0时，才说明都检测到了有效目标
        if left_list_size > 0 and right_list_size > 0 and detect_ret_left and detect_ret_right:
            # 此时可以获取左相机中最大的目标的bounding box信息，包括中心点坐标(cx, cy)、宽度(height)、高度(width)等
            # [0]表示x坐标，[1]表示y坐标，[2]表示宽度，[3]表示高度，可参见object_detection.py中的bounding_box_list的定义
            left_cx = detect_left.bounding_box_list[left_max][0]
            left_cy = detect_left.bounding_box_list[left_max][1]
            left_width = detect_left.bounding_box_list[left_max][2]
            left_height = detect_left.bounding_box_list[left_max][3]

            # 同理，获取右相机中最大目标的bounding box信息
            right_cx = detect_right.bounding_box_list[right_max][0]
            right_cy = detect_right.bounding_box_list[right_max][1]
            right_width = detect_right.bounding_box_list[right_max][2]
            right_height = detect_right.bounding_box_list[right_max][3]

            # 分别获取左、右相机中目标物体的检测框的左上角顶点坐标
            left_x1 = int(left_cx - left_width / 2)
            left_y1 = int(left_cy - left_height / 2)
            right_x1 = int(right_cx - right_width / 2)
            right_y1 = int(right_cy - right_height / 2)

            """
            # 将检测框的局部画面裁切出来
            left_bbox_roi = split_image.extract_image(left_frame, left_x1, left_y1, left_width, left_height)
            right_bbox_roi = split_image.extract_image(right_frame, right_x1, right_y1, right_width, right_height)

            # 构造特征点匹配对象
            matcher = object_detection.MatchFeature(left_bbox_roi, right_bbox_roi)

            # 执行ORB特征点匹配
            matcher.do_orb_match()

            # 默认检测距离0.0000m
            distance = 0.0000
            """

            # 若匹配成功，则计算距离
            if True:
                # 传入该物体在两个相机画面对应各自画面的坐标
                # 双目相机基线为 60 mm，即 0.06 m；每个相机镜头的焦距为 3.0 mm，即 0.003 m
                stereo_dist_calculator = distance_estimate.stereo_distance_calc((left_cx, left_cy),
                                                                                (right_cx, right_cy), 0.060, 0.003)
                distance = stereo_dist_calculator.calc_distance()

                # 显示左、右相机中检测到目标并标注后的图像，以及特征点匹配后的图像
                cv2.imshow("Left", detect_left.labeled_image)
                cv2.imshow("Right", detect_right.labeled_image)
                # cv2.imshow("Matches", matcher.get_matched_image())

                # 将距离单位转换为mm
                info.distance = distance * 1000

                # 取bounding box中垂线上与bounding box上沿相交的点
                # 此时只是发布了目标在图像上的中心点坐标，并非实际的三维坐标，后续需要根据相机标定数据进行进一步处理
                info.cx = left_cx
                info.cy = left_y1
                info.cz = 0
                info.height = 0
                info.width = 0

                # 发布消息，保持10Hz的发布频率
                publisher.publish(info)
                rate.sleep()

                # 打印目标距离
                print(
                    f"目标（左：({left_cx}, {left_cy}, {left_width}, {left_height})，右：({right_cx}, {right_cy}, {right_width}, {right_height})的距离为：{distance} m")
            else:
                cv2.imshow("Left", detect_left.labeled_image)
                cv2.imshow("Right", detect_right.labeled_image)
                print("未检测到合适的目标")

        # 判断是否按下了键盘上的“Q”键，若按下则退出循环
        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('q') or pressed_key == ord('Q'):
            break

    # 释放摄像头与关闭窗口
    camera.release()
    cv2.destroyAllWindows()
    print("主程序退出")


# 主函数入口判别
if __name__ == '__main__':
    main()
