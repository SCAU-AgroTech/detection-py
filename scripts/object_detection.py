#! /usr/bin/env python

import cv2
import numpy as np
from pathlib import Path  # 导入Path用于判断路径是否有效

# 默认图像的宽度
INPUT_WIDTH: int = 640

# 默认图像的高度
INPUT_HEIGHT: int = 640

# 默认的分数阈值
SCORE_THRESHOLD: float = 0.55

# 默认的非极大值抑制阈值
NMS_THRESHOLD: float = 0.45

# 默认的YOLO模型的维度
MODEL_DIMENSIONS: int = 85

# 默认的YOLO模型的行数
YOLO_ROWS: int = 25200

# 默认的字体缩放比例
FONT_SCALE: float = 0.70

# 默认的缩放因子
SCALE_FACTOR: float = 1.0 / 255.0

# 默认的字体
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX

# 默认的字体厚度
THICKNESS: int = 1

# 使用Tuple储存颜色，使用BGR颜色空间
# 黑色
BLACK: tuple = (0, 0, 0)

# 蓝色
BLUE: tuple = (255, 178, 50)

# 红色
RED: tuple = (0, 0, 255)

# 黄色
YELLOW: tuple = (0, 255, 255)


# Bounding box的类
class BoundingBox:
    # 中心点的x坐标
    x: int

    # 中心点的y坐标
    y: int

    # 检测框的高度
    height: int

    # 检测框的宽度
    width: int

    # 目标框的序号
    index: int


# 该类主要用于目标检测
class ObjectDetection:
    # 经过标注后的图像
    labeled_image: np.ndarray = None

    # 单幅画面中所有的检测框的列表，每一个元素都是一个通过tuple或者Numpy数组表示的BoundingBox对象
    # 注意，每一个bounding box的元素的[0]为x坐标，[1]为y坐标，[2]为宽度，[3]为高度，并非与np.ndarray的x,y,h,w顺序相同
    bounding_box_list: list = []

    # 模型的绝对路径
    __model_path: str = None

    # 图像的绝对路径
    # __image_path: str = None

    # 图像的矩阵
    __image_matrix: np.ndarray = None

    # 类别名称列表
    __class_names: list = []

    # 网络模型对象
    __model_obj: cv2.dnn.Net = None

    # 指示是否从绝对路径加载了图像
    __is_image_loaded_from_path: bool = False

    # 该私有函数用于在特定图像上绘制边框，并且在bounding box的顶部绘制文字（ASCII字符集）
    def __draw_label(self, input_image: np.ndarray, label: str, x1: int, y1: int, x2: int, y2: int):
        try:
            # 在bounding box的顶部绘制文字
            label_size: tuple = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
            y1 = max(y1, label_size[1])

            # 左上角顶点坐标
            top_left_point: tuple = (x1, y1 - label_size[0][1])

            # 右下角顶点坐标
            bottom_right_point: tuple = (x2, y2)

            print(bottom_right_point)

            # 绘制黄色矩形
            cv2.rectangle(input_image, top_left_point, bottom_right_point, YELLOW, 2)

            # 在矩形框上绘制文字
            cv2.putText(input_image, label, (x1, y1 + label_size[0][1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS)

            # 指示绘制成功
            return True
        except Exception as draw_label_exception:
            # 反馈报错信息
            print('在绘制标签时出现错误：' + str(draw_label_exception))
            return False

    # 加载神经网络及其初始化
    def __get_net(self):
        try:
            # 加载Torch训练而得的神经网络
            net = cv2.dnn.readNetFromONNX(self.__model_path)

            # 返回神经网络对象
            return net
        except Exception as load_net_exception:
            # 反馈报错信息
            print('在加载神经网络时出现错误：' + str(load_net_exception))
            exit(-1)

    # 判断类内部的图像矩阵是否已经加载
    def __is_image_matrix_loaded(self):
        try:
            # 判断图像矩阵是否为空
            return self.__image_matrix is not None
        except Exception as is_image_matrix_loaded_exception:
            # 反馈报错信息
            print('在判断图像矩阵是否已经加载时出现错误：' + str(is_image_matrix_loaded_exception))
            exit(-1)

    # 判断类内部的模型路径是否为有效路径
    def __is_model_path_invalid(self):
        try:
            # 判断模型路径是否有效
            file_path = Path(self.__model_path)

            # 从Path对象中获取是否存在
            return file_path.exists()
        except Exception as is_model_path_invalid_exception:
            # 反馈报错信息
            print('在判断模型路径是否有效时出现错误：' + str(is_model_path_invalid_exception))
            exit(-1)

    # 判断类内部的网络模型对象是否已经加载
    def __is_model_obj_loaded(self):
        try:
            # 判断网络模型对象是否为空
            return self.__model_obj is not None
        except Exception as is_model_obj_loaded_exception:
            # 反馈报错信息
            print('在判断网络模型对象是否已经加载时出现错误：' + str(is_model_obj_loaded_exception))
            exit(-1)

    # 在进行前向推理前对图像进行预处理
    def __pre_process(self, input_image: np.ndarray, model):
        try:
            # 从输入图像创建一个blob
            blob = cv2.dnn.blobFromImage(input_image, SCALE_FACTOR, (INPUT_WIDTH, INPUT_HEIGHT), (0, 0, 0), True,
                                         crop=False)

            # 设置模型的输入
            model.setInput(blob)

            # 默认使用GPU进行推理加速
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # 指示预处理成功
            return True
        except Exception as pre_process_exception:
            # 反馈报错信息
            print('在预处理图像时出现错误：' + str(pre_process_exception))
            exit(-1)

    # 执行前向推理
    def __detect_forward(self, model):
        try:
            # 执行前向推理
            output_mat_vect = model.forward(model.getUnconnectedOutLayersNames())

            # 返回推理结果
            return output_mat_vect
        except Exception as detect_forward_exception:
            # 反馈报错信息
            print('在执行前向推理时出现错误：' + str(detect_forward_exception))
            exit(-1)

    # 对推理数据进行读取暂存，并对多重相近且重叠的bounding box进行NMS非最大抑制的过滤处理
    # 读取torch模型的核心函数
    def __post_process(self, input_image: np.ndarray, detection_results: np.ndarray):
        """
        说明：

         *    这个函数返回的对象是一个二维数组，输出取决于输入的大小。例如，当默认输入大小为 640 时，我们得到一个大小为 25200 × 85（行和列）的 2D 矩阵
         *    行表示检测次数
         *    因此，每次网络运行时，它都会预测 25200 个边界框。每个边界框都有一个包含 85 个条目的一维数组，用于说明检测的质量，此信息足以筛选出所需的检测

         *    数据的排列大概长这样：
         *        X   |   Y   |   W   |   H   |   目标置信度（Confidence）  |   （最多）80个类别物体的得分权重
         *    ^~~~~~~~^~~~~~~~^~~~~~~~^~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *    数据第0位 数据第1位 数据第2位 数据第3位         数据第4位                                 数据第5-85位

         *    网络根据 blob 的输入大小（默认 640 × 640）生成输出坐标。因此，应将坐标乘以调整大小系数以获得实际输出
        """

        # 类别ID的列表集合
        class_ids: list = []

        # 存储检测框的列表
        confidences: list = []

        # 存储检测框的列表
        boxes: list = []

        # 获取推理结果矩阵的行数
        result_rows: int = detection_results[0].shape[1]

        # 获取输入图像的高度与宽度
        image_height, image_width = input_image.shape[:2]

        # 计算缩放比例因子
        x_factor: float = image_height / INPUT_HEIGHT
        y_factor: float = image_width / INPUT_WIDTH

        # 迭代遍历推理结果
        for row_num in range(result_rows):
            # 获取每一行的数据
            row_data = detection_results[0][0][row_num]

            # 获取置信度
            confidence: float = float(row_data[4])

            # 过滤掉低于阈值的检测结果：如果置信度大于阈值
            if confidence > SCORE_THRESHOLD:
                # 拿到所有的ID数据分数
                class_score: np.ndarray = row_data[5:]

                # 获取最高分的ID
                max_class_id = np.argmax(class_score)

                # 若完成上述步骤，则继续获取细致的检测框信息
                if class_score[max_class_id] > SCORE_THRESHOLD:
                    # 向confidences列表中追加置信度
                    confidences.append(confidence)

                    # 向class_ids列表中追加类别ID
                    class_ids.append(max_class_id)

                    # 获取检测框的坐标与尺寸信息
                    cx: int = int(row_data[0])
                    cy: int = int(row_data[1])
                    w: int = int(row_data[2])
                    h: int = int(row_data[3])

                    # 计算尺寸在原图像中的坐标，其中，x1与y1为左上角坐标
                    x1: int = int((cx - w / 2) * x_factor)
                    y1: int = int((cy - h / 2) * y_factor)
                    height = int(h * y_factor)
                    width = int(w * x_factor)

                    # 创建一个BoundingBox对象，使用numpy数组存储检测框的坐标与尺寸信息
                    # 已弃用BoundingBox类
                    box = np.array([x1, y1, width, height])

                    # 向boxes列表中追加检测框
                    boxes.append(box)

        # 在完成读取信息后，还需要进行NMS非极大值抑制处理，避免出现大量重叠的检测框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

        # 遍历检测框的索引，筛选出最终的检测结果
        for i in indices:
            bbox: list = boxes[i]

            # 获取检测框的坐标与尺寸信息，其中x1与y1为左上角坐标
            x1: int = bbox[0]
            y1: int = bbox[1]
            height: int = bbox[2]
            width: int = bbox[3]

            # 更新类的bounding_box_list属性
            self.bounding_box_list.append((x1, y1, width, height))

            # 获取标签信息
            label = "{}:{:.2f}".format(self.__class_names[class_ids[i]], confidences[i])

            # 绘制检测框与标签
            self.__draw_label(input_image, label, x1, y1, x1 + width, y1 + height)

        # 更新类的labeled_image属性，并返回最终的图像
        self.labeled_image = input_image
        return input_image

    # 构造函数，需要传入模型的绝对路径，图像矩阵以及包含训练时设定的所有标签名称的列表
    def __init__(self, model_path: str, image_matrix: np.ndarray, class_names: list, net_obj: cv2.dnn.Net):
        # 初始化赋值
        self.labeled_image = np.array([])
        self.bounding_box_list = []
        self.__model_obj = net_obj

        self.__model_path = model_path
        self.__image_matrix = image_matrix
        self.__class_names = class_names
        self.__is_image_loaded_from_path = False

    # 核心函数，用于实现目标识别检测与数据处理的主要函数
    def detect(self):
        try:
            # 先判空
            if not (self.__is_image_matrix_loaded() and self.__is_model_path_invalid() and self.__is_model_obj_loaded()):
                raise Exception("图像矩阵或模型未能正确加载")

            # 加载模型
            model: cv2.dnn.Net = self.__model_obj

            # 图像预处理
            self.__pre_process(self.__image_matrix, model)

            # 获取推理矩阵
            result_mat_array: np.ndarray = self.__detect_forward(model)

            # 获取推理数据并拿到处理后的图像
            self.labeled_image = self.__post_process(self.__image_matrix, result_mat_array)

            cv2.imshow("Actual", self.labeled_image)
            # 返回
            return True

        except Exception as detect_exception:
            print('在目标检测.检测核心.shell的过程中出现错误：' + str(detect_exception))
            exit(-1)

    # 该函数用于获取目标框中最大的bounding box的索引
    def get_max_box_index(self):
        # 临时索引，默认从0开始
        tmp_index: int = 0
        tmp_sum: int = 0

        # 遍历bounding_box_list列表
        for i in range(len(self.bounding_box_list)):
            # 每一个bounding box的元素的[0]为x坐标，[1]为y坐标，[2]为宽度，[3]为高度
            if ((tmp_sum < self.bounding_box_list[i][2] + self.bounding_box_list[i][3]) & (
                    self.bounding_box_list[i][0] <= INPUT_WIDTH & self.bounding_box_list[i][1] <= INPUT_HEIGHT)):
                tmp_sum = self.bounding_box_list[i][2] + self.bounding_box_list[i][3]
                tmp_index = i

        return tmp_index

    # 析构函数
    def __del__(self):
        pass


# 该类主要用于实现匹配两张图像上的特征点并维护特征点计数
class MatchFeature:
    # 该值指示是否匹配
    is_matched: bool = False

    # 指示H单应性矩阵是否有效
    is_h_matrix_valid: bool = False

    # 左、右图像以及匹配后的图像
    __left_image: np.ndarray = None
    __right_image: np.ndarray = None
    __matched_image: np.ndarray = None

    # 该函数用于进行ORB特征点检测
    def do_orb_match(self, distance_threshold: float = 30.0, good_matches_threshold: int = 20, h_matrix_threshold: int = 20):
        left_image = self.__left_image
        right_image = self.__right_image

        # 创建ORB特征检测器
        orb_detector = cv2.ORB_create()

        # 检测关键点并计算描述符
        left_key_points, left_descriptors = orb_detector.detectAndCompute(left_image, np.array([]))
        right_key_points, right_descriptors = orb_detector.detectAndCompute(right_image, np.array([]))

        # 匹配描述符
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 获取匹配结果
        matches: list = bf_matcher.match(left_descriptors, right_descriptors)

        # 对匹配结果进行排序，并获取最大的与最小的汉明距离
        # matches.sort(key=lambda x: x.distance, reverse=False)
        # maximum_distance = matches[-1].distance
        # minimum_distance = matches[0].distance

        # 筛选良好的匹配，遍历所有匹配结果并计算汉明距离
        good_matches: list = []
        for match in matches:
            if match.distance < distance_threshold:
                good_matches.append(match)

        # 判断是否匹配
        self.is_matched = len(good_matches) >= good_matches_threshold

        # 计算两张图片中匹配的描述符并进行连线
        cv2.drawMatches(left_image, left_key_points, right_image, right_key_points, good_matches, self.__matched_image)

        # 使用RANSAC算法计算H矩阵
        left_scenes: np.ndarray = np.zeros((len(good_matches), 2), dtype=np.float32)
        right_scenes: np.ndarray = np.zeros((len(good_matches), 2), dtype=np.float32)
        for match in good_matches:
            left_scenes[match, :] = left_key_points[match.queryIdx].pt
            right_scenes[match, :] = right_key_points[match.trainIdx].pt

        # 获取单应性矩阵
        h_matrix, out_mask = cv2.findHomography(left_scenes, right_scenes, cv2.RANSAC, 3.0)
        valid_h_matrix_num = cv2.countNonZero(out_mask)
        self.is_h_matrix_valid = valid_h_matrix_num >= h_matrix_threshold

    # 该函数用于获取经过ORB特征匹配后连线的图像
    def get_matched_image(self):
        return self.__matched_image

    # 构造函数，请传入左、右相机画面需要对比的区域的图像
    # 请勿传入空值或者是双目相机的两个原始图像，这会导致意料不到的错误
    def __init__(self, left_image: np.ndarray, right_image: np.ndarray):
        # 先判空
        if left_image is None or right_image is None:
            raise Exception("进行特征点匹配时，左、右图像不能为空")

        # 将图像转换为灰度图
        self.__left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        self.__right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # 初始化其他属性
        self.is_matched = False
        self.is_h_matrix_valid = False

    # 析构函数
    def __del__(self):
        pass
