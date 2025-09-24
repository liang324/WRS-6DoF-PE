#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：WRS-6DoF-PE
@File    ：simulation_utils.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""
import pyransac3d as pyrsc
import copy
import datetime
import glob
import json
import logging
import pickle
import sys
import time

import matplotlib.pyplot as plt

from robot_grasp.planar_registration import robust_plane_alignment, icp_refinement, visualize_registration, \
    adaptive_plane_alignment
from YOLOv9_Detect_API import DetectAPI
from camera_utils import capture_point_cloud
import shutil
import os
import torch
import cv2
import numpy as np
from pathlib import Path

from robot_grasp.robot_grasping_opop import RobotGrasping
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d
from typing import List, Tuple
import os.path
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from typing import Tuple, List, Dict
from dataclasses import dataclass

import modeling.collision_model as cm
import visualization.panda.world as wd
import modeling.geometric_model as gm
from human_grasping import find_robot_grabbable_positions, find_best_human_grasp_position, simulate_grasp
# from robot_grasp_opop import RobotGrasping
import robot_sim.robots.gofa5.gofa5_Ag145 as gf5

OBJECT_MODEL_RACK = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack510.STL"
TUBE_TYPE_CONFIG = {
    "TubeO": {  # 橙色试管
        "model_path": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\zheguang_tube_quan_center.STL",
        "color": [1.0, 0.65, 0.0, 1.0]
    },
    "TubeB": {  # 蓝色试管
        "model_path": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\zheguang_tube_quan_center.STL",
        "color": [0.0, 0.0, 1.0, 1.0]
    },
    "TubeG": {  # 绿色试管
        "model_path": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\zheguang_tube_quan_center.STL",
        "color": [0.56, 0.93, 0.56, 1.0]
    },
    # 如果 JSON 中 label 无法识别则使用默认
    "default": {
        "model_path": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\zheguang_tube_quan_center.STL",
        "color": [0.8, 0.8, 0.8, 1.0]
    }
}

# 获取当前时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'./log_filename/grasping_log_detection_{timestamp}.txt'

# 配置日志记录
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class_names = {
    0: "rack510",
    1: "TubeB",
    2: "TubeG",
    3: "TubeO",
}

sam_checkpoint = r"D:\AI\ABB_wrs_hu_sam\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"

save_folder = r'D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\realsense_data_rack510'
save_dir_pcd = os.path.join(save_folder, 'pcd')  # 用于保存原始图像点云的文件夹
mask_output_folder = os.path.join(save_folder, 'masks')  # 用于保存掩码图像的文件夹
yolo_detection_image_path = os.path.join(save_folder, 'yolo_detections.jpg')  # 保存 YOLO 检测结果的图像路径
point_cloud_output_folder = os.path.join(save_folder, 'point_clouds')  # 用于保存点云文件的文件夹
plane_output_folder = os.path.join(save_folder, 'planes')  # 用于保存提取的平面点云的文件夹
remaining_output_folder = os.path.join(save_folder, 'remaining')  # 用于保存剩余点云的文件夹
registration_output_folder = os.path.join(save_folder, 'registration')  # 用于保存配准结果的文件夹


def clear_folder(folder_path):
    """
    清空指定文件夹下的所有文件和子文件夹
    """
    try:
        if os.path.exists(folder_path):  # 检查文件夹是否存在
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            print(f"成功清空文件夹: {folder_path}")
        else:
            print(f"文件夹不存在 ,跳过清空: {folder_path}")
    except Exception as e:
        print(f"清空文件夹 {folder_path} 失败: {e}")


def convert_units(self, pcd_path):
    """
    Converts the units of a point cloud from millimeters to meters
    """
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        points_meters = points / 1000.0
        pcd.points = o3d.utility.Vector3dVector(points_meters)
        output_path = pcd_path.replace(".ply", "_meters.ply")
        return pcd

        # o3d.io.write_point_cloud(output_path, pcd)
        # print(f"Converted units and saved to: {output_path}")

    except Exception as e:
        print(f"Error converting units for {pcd_path}: {e}")
        return None  # Indicate failure


def extract_plane(pcd_path, save_dir_plane, save_dir_remaining, threshold=0.001, max_iterations=10000):
    """
    从点云中提取平面并保存
    :param pcd_path: 点云路径
    :param save_dir_plane: 存路径
    :param save_dir_remaining: 剩余点云保存路径
    :param threshold: RANSAC分割阈值
    :param max_iterations: RANSAC最大迭代次数
    """

    # 1. 创建保存目录
    if not os.path.exists(save_dir_plane):
        os.makedirs(save_dir_plane)
    if not os.path.exists(save_dir_remaining):
        os.makedirs(save_dir_remaining)

    # 2. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 3. 平面检测与分割
    plane = pyrsc.Plane()
    try:
        plane_model, inliers = plane.fit(points, thresh=threshold, maxIteration=max_iterations)
        # print("plane_model, inliers", plane_model, inliers)
    except Exception as e:
        print(f"Error during plane fitting: {e}")
        print("Returning empty point clouds.")
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()  # 返回空点云

    # 4. 分割点云
    if len(inliers) > 0:
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
    else:
        print("No inliers found. Returning empty point clouds.")
        inlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud = pcd  # 所有点都是外点

    # 5. 保存点云
    filename = os.path.basename(pcd_path).split(".")[0]
    plane_filename = os.path.join(save_dir_plane, f"{filename}_plane.ply")
    remaining_filename = os.path.join(save_dir_remaining, f"{filename}_remaining.ply")

    o3d.io.write_point_cloud(plane_filename, inlier_cloud)
    o3d.io.write_point_cloud(remaining_filename, outlier_cloud)

    # print(f"平面点云保存至: {plane_filename}")
    # print(f"剩余点云保存至: {remaining_filename}")
    return inlier_cloud, outlier_cloud, plane_filename, remaining_filename


def sam_segment(img0, bbox):
    # 初始化 SAM 模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 根据模型类型加载 SAM 模型
    predictor = SamPredictor(sam)  # 创建 SAM 预测器

    # 转换图像颜色空间
    image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB
    predictor.set_image(image)  # 设置 SAM 预测器的图像

    # 将 YOLO 边界框传递给 SAM 进行分割
    masks, _, _ = predictor.predict(
        point_coords=None,  # 没有点提示
        point_labels=None,  # 没有点标签
        box=np.array(bbox),  # 使用边界框作为提示
        multimask_output=False  # 只输出一个掩码
    )
    return masks[0]


def save_object_mask_on_black_background(img, masks, bboxes, labels, save_folder):
    """
    保存每个检测到的物体的掩码图像,物体显示为白色,背景为黑色,并保持原始图像大小

    Args:
        img (numpy.ndarray): 原始图像
        masks (list): SAM 生成的掩码列表
        bboxes (list): YOLOv9 检测到的边界框列表
        labels (list): 物体标签列表
        save_folder (str): 保存掩码图像的文件夹路径
    """
    os.makedirs(save_folder, exist_ok=True)

    # for i, (mask, bbox, label) in enumerate(zip(masks, bboxes, labels)):
    # 创建一个与原始图像大小相同的黑色图像
    mask_img = np.zeros_like(img, dtype=np.uint8)

    # 将掩码区域设置为白色
    mask_area = masks > 0
    mask_img[mask_area] = [255, 255, 255]  # 白色

    # 保存掩码图像
    save_path = os.path.join(save_folder, f"mask_{labels}.png")  # 使用物体名称作为文件名
    cv2.imwrite(save_path, mask_img)
    print(f"物体 {labels} 的掩码图像已保存: {save_path}")
    return mask_img, save_path


def save_tube_object_mask_on_black_background_op(img, mask, label_name, counter_idx, save_folder):
    """
    保存每个检测到的物体的掩码图像,物体显示为白色,背景为黑色,并保持原始图像大小
    Args:
        img: 原始图像 (H, W, 3)
        mask: SAM 生成的单个物体掩码 (H, W) 或 (H, W, 1)
        label_name: 该物体的标签 (str)
        counter_idx: 该标签的编号 (int)
        save_folder: 保存路径
    """
    os.makedirs(save_folder, exist_ok=True)

    # 适配 mask 的维度
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_area = mask > 0

    # 创建黑色背景
    mask_img = np.zeros_like(img, dtype=np.uint8)

    # 设置白色的掩码部分
    mask_img[mask_area] = [255, 255, 255]

    # 文件名
    save_path = os.path.join(save_folder, f"mask_{label_name}_{counter_idx}.png")

    # 保存掩码图像
    cv2.imwrite(save_path, mask_img)
    print(f"物体 {label_name}_{counter_idx} 的掩码图像已保存: {save_path}")

    return mask_img, save_path


def save_yolo_detection_image(img, detections, detect_api, save_path, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 YOLO 检测到的边界框和标签 ,并保存图像

    Args:
        img (numpy.ndarray): 原始图像
        detections (list): YOLOv9 检测结果列表 (xyxy 格式)
        detect_api: 检测 API 对象 (必须有 model.names 属性)
        save_path (str): 保存图像的路径
        color (tuple): 边界框的颜色 (BGR 格式)
        thickness (int): 边界框的粗细
    """
    img_copy = img.copy()
    for *xyxy, conf, cls in detections:  # 解包检测结果
        x0, y0, x1, y1 = map(int, xyxy)  # 转换为整数
        class_name = detect_api.names[int(cls)]  # 获取类别名称
        confidence = float(conf)

        cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, thickness)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img_copy, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    cv2.imwrite(save_path, img_copy)
    print(f"带有 YOLO 检测结果的图像已保存: {save_path}")


# ----------------- 从掩码和深度图像中提取点云 -----------------
def extract_point_cloud(color_image, mask_image, depth_image, fx, fy, cx, cy, depth_scale=0.001):
    """
    从掩码图像和深度图像中提取点云,并保存原图颜色信息

    Args:
        color_image (numpy.ndarray): 彩色图像
        mask_image (numpy.ndarray): 掩码图像,白色物体,黑色背景
        depth_image (numpy.ndarray): 深度图像
        fx (float): 相机焦距 x
        fy (float): 相机焦距 y
        cx (float): 相机中心 x
        cy (float): 相机中心 y
        depth_scale (float): 深度比例因子,用于将深度图像中的像素值转换为以米为单位的深度值

    Returns:
        open3d.geometry.PointCloud: 点云对象
    """
    points = []
    colors = []  # 存储颜色信息
    for v in range(mask_image.shape[0]):
        for u in range(mask_image.shape[1]):
            if mask_image[v, u][0] == 255:  # 确保是白色像素
                z = depth_image[v, u] * depth_scale  # 深度信息从深度图像单位转换为米
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

                # 获取颜色信息
                b, g, r = color_image[v, u]  # OpenCV 图像颜色通道顺序为 BGR
                colors.append([r / 255.0, g / 255.0, b / 255.0])  # 转换为 0-1 范围

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))  # 设置颜色信息
    return pcd


def convert_units(pcd_path, output_path=None):
    """
    Converts the units of a point cloud from millimeters to meters.
    """
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        points_meters = points / 1000.0
        pcd.points = o3d.utility.Vector3dVector(points_meters)

        # Determine the output path for the converted point cloud
        if output_path is None:
            converted_meters_pcd_path = pcd_path.replace(".ply", "_meters.ply")
        else:
            converted_meters_pcd_path = output_path

        o3d.io.write_point_cloud(converted_meters_pcd_path, pcd)
        print(f"Converted units and saved to: {converted_meters_pcd_path}")
        return pcd, converted_meters_pcd_path

    except Exception as e:
        print(f"Error converting units for {pcd_path}: {e}")
        return None, None  # Indicate failure


def icp_plane_estimate_pose_rack_tube(label_name, plane_cloud, template_cloud_rack):
    """
    估计场景中物体的位姿,并保存配准结果

    Args:
        label (str): 物体标签,例如 "t 0.95" (tube) 或 "b 0.88" (rack)
        plane_cloud (open3d.geometry.PointCloud): 从场景中提取的平面点云
        template_cloud_rack (open3d.geometry.PointCloud):  试管架模板点云

    Returns:
        numpy.ndarray:  初始变换矩阵
    """

    # 选择模板点云
    if label_name == "rack510":
        template_cloud = template_cloud_rack  # 注意这里,试管用的是试管架的模板
    else:
        print(f"未知的类别: {label_name}")
        return None  # 或者抛出异常

    # 确保点云有法向量
    if not plane_cloud.has_normals():
        plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    if not template_cloud.has_normals():
        template_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    # 阶段1: 平面法向量对齐
    print("\n=== 阶段 1: 平面法向量对齐 ===")
    # initial_transformation = robust_plane_alignment(plane_cloud, template_cloud)
    initial_transformation = adaptive_plane_alignment(plane_cloud, template_cloud, noise_level="medium")
    final_transformation = initial_transformation
    print("初始变换矩阵:\n", final_transformation)
    visualize_registration(plane_cloud, template_cloud, final_transformation, "Initial Alignment")

    # 阶段2: ICP精配准 (仅对试管进行精配准)
    if label_name == "rack510":  # 试管进行ICP精配准
        print("\n=== 阶段 2: ICP精配准 ===")
        final_transformation = icp_refinement(plane_cloud, template_cloud, initial_transformation)
        print("精配准变换矩阵:\n", final_transformation)
        visualize_registration(plane_cloud, template_cloud, final_transformation, "Final Registration")

    return final_transformation


def icp_plane_estimate_pose(plane_cloud,
                            template_cloud,
                            do_icp_refine=True):
    """
    估计场景中物体的位姿,并保存配准结果

    Args:
        plane_cloud (open3d.geometry.PointCloud): 提取的平面点云
        template_cloud (open3d.geometry.PointCloud): 模板点云
        do_icp_refine (bool): 是否进行 ICP 精配准

    Returns:
        numpy.ndarray: 4x4 初始或精配准变换矩阵
    """
    # 确保有法向量
    if not plane_cloud.has_normals():
        plane_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
    if not template_cloud.has_normals():
        template_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )

    # 阶段1: 平面法向量对齐
    print("\n=== 阶段 1: 平面法向量对齐 ===")
    initial_transformation = adaptive_plane_alignment(plane_cloud, template_cloud, noise_level="medium")
    final_transformation = initial_transformation
    print("初始变换矩阵:\n", final_transformation)
    visualize_registration(plane_cloud, template_cloud, final_transformation, "Initial Alignment")

    # 阶段2: ICP 精配准（可选）
    if do_icp_refine:
        print("\n=== 阶段 2: ICP 精配准 ===")
        final_transformation = icp_refinement(plane_cloud, template_cloud, initial_transformation)
        print("精配准变换矩阵:\n", final_transformation)
        visualize_registration(plane_cloud, template_cloud, final_transformation, "Final Registration")

    return final_transformation


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    计算两个边界框的 IOU (Intersection over Union)
    """
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    intersect_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersect_area

    iou = intersect_area / union_area if union_area > 0 else 0
    return iou


from typing import List, Tuple, Optional
import numpy as np


def select_largest_object(detections_tensor: List[np.ndarray],
                          detect_api,
                          target_label: str,
                          check_rotate: bool = False
                          ) -> Tuple[
    Optional[np.ndarray], bool, Optional[Tuple[float, float, float, float]], Optional[float], str]:
    """
    从 YOLOv9 检测结果中选择面积最大的目标对象

    Args:
        detections_tensor: YOLOv9 检测结果张量列表
        detect_api: 检测 API 对象 (必须有 model.names 属性)
        target_label: 目标类别标签
        check_rotate: 是否根据宽高比判断需要旋转矩阵

    Returns:
        largest_object: 最大目标的检测结果 (未检测到则为 None)
        rotate_matrix: 是否需要旋转矩阵
        bbox: 目标边界框 (x1, y1, x2, y2)
        confidence: 检测置信度
        label_name: 类别名称
    """
    largest_object = None
    max_area = 0
    rotate_matrix = False
    bbox = None
    best_confidence = None
    label_name = target_label

    if detections_tensor is not None and len(detections_tensor) > 0:
        for det in detections_tensor[0]:
            x1, y1, x2, y2, conf, label = det.tolist()
            # current_label = class_names.get(int(label), "Unknown")
            current_label = detect_api.names[int(label)]

            if current_label == target_label:
                width = x2 - x1
                height = y2 - y1
                area = width * height

                if area > max_area:
                    max_area = area
                    largest_object = det
                    rotate_matrix = (width < height) if check_rotate else False
                    bbox = (x1, y1, x2, y2)
                    best_confidence = conf

    return largest_object, rotate_matrix, bbox, best_confidence, label_name


def extract_tube_point_clouds(image: np.ndarray,
                              depth_image: np.ndarray,
                              detections_tensor: List[np.ndarray],
                              tube_labels: List[str],
                              fx: float, fy: float, cx: float, cy: float,
                              point_cloud_output_folder: str,
                              mask_output_folder: str,
                              plane_output_folder: str,
                              remaining_output_folder: str):
    """
    从 YOLO 检测结果中提取指定标签的试管点云
    返回:
        image: 原始图像
        all_tube_labels: {tube_id: label}
        all_tube_confidences: {tube_id: confidence}
        all_point_clouds: list[o3d.geometry.PointCloud]
    """
    print("\n[INFO] === 开始执行 extract_tube_point_clouds() ===")
    start_time = time.time()

    label_counter = {}
    tube_data_list = []

    for det in detections_tensor[0]:
        x1, y1, x2, y2, confidence, label = det.tolist()
        label_name = class_names.get(int(label), "unknown")
        print(f"[INFO] 检测标签: {label_name}, 置信度: {confidence:.3f}")

        if label_name in tube_labels:
            # 标签计数
            label_counter[label_name] = label_counter.get(label_name, 0) + 1
            tube_id = f"{label_name}_{label_counter[label_name]}"
            # 边界框
            bbox_tube = (int(x1), int(y1), int(x2), int(y2))
            print(f"[INFO] Tube ID: {tube_id}, 边界框: {bbox_tube}")

            # SAM 分割
            t0 = time.time()
            mask = sam_segment(image, bbox_tube)
            print(f"[INFO] SAM 分割完成,用时 {time.time() - t0:.3f} 秒")

            # 保存 mask
            mask_image, _ = save_tube_object_mask_on_black_background_op(
                image, mask, label_name, label_counter[label_name], mask_output_folder
            )
            print(f"[INFO] Mask 已保存到 {mask_output_folder},用时 {time.time() - t0:.3f} 秒")

            # 点云提取
            t0 = time.time()
            point_cloud = extract_point_cloud(image, mask_image, depth_image, fx, fy, cx, cy)
            # 保存原始点云
            point_cloud_path = os.path.join(point_cloud_output_folder, f"tube_{tube_id}.ply")
            o3d.io.write_point_cloud(point_cloud_path, point_cloud)
            print(f"[INFO] 点云已保存到: {point_cloud_path},用时 {time.time() - t0:.3f} 秒")

            # 平面分割
            t0 = time.time()
            plane_cloud, remaining_cloud, plane_filename, remaining_filename = extract_plane(point_cloud_path,
                                                                                             plane_output_folder,
                                                                                             remaining_output_folder,
                                                                                             threshold=0.002,
                                                                                             max_iterations=10000)

            print(f"物体 {tube_id} 的平面点云已保存: {plane_filename}")
            print(f"物体 {tube_id} 的剩余点云已保存: {remaining_filename}")

            # 保存到列表
            tube_data_list.append({
                "tube_id": tube_id,
                "label": label_name,
                "confidence": float(confidence),
                "pcd": plane_cloud,
                "plane_path": plane_filename
            })

    return tube_data_list


@dataclass
class TubeProjectionResult:
    """试管投影结果"""
    tube_name: str
    projected_center: np.ndarray
    convex_hull: np.ndarray
    area: float
    grid_position: Tuple[int, int]
    overlap_score: float
    label: str = None
    confidence: float = None


class SimplifiedTubePoseEstimator:
    """试管位姿估计器 """

    def __init__(self, rack_shape=(5, 10),
                 center_holes=[(2, 4), (2, 5)],
                 tube_spacing=0.025,
                 tube_diameter=0.021):
        self.RACK_SHAPE = rack_shape
        self.CENTER_HOLES = center_holes
        self.TUBE_SPACING = tube_spacing
        self.TUBE_DIAMETER = tube_diameter
        self.rack_normal = None
        self.rack_center = None
        self.rack_u_axis = None
        self.rack_v_axis = None

    def compute_rack_plane_info(self, rack_pcd: o3d.geometry.PointCloud):
        print("计算试管架平面信息...")

        # 1. 计算 OBB
        obb = rack_pcd.get_oriented_bounding_box()

        # 2. 获取旋转矩阵
        rotation_matrix = obb.R

        # 3. 提取法向量 / 中心点 / U 轴 / V 轴
        self.rack_normal = rotation_matrix[:, 2]  # Z 轴（法向量）
        self.rack_center = obb.center
        self.rack_u_axis = rotation_matrix[:, 0]  # X 轴
        self.rack_v_axis = rotation_matrix[:, 1]  # Y 轴

        print(f"平面法向量: {self.rack_normal}")
        print(f"平面中心: {self.rack_center}")
        print(f"U轴: {self.rack_u_axis}")
        print(f"V轴: {self.rack_v_axis}")

        obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        # 将坐标系平移旋转到 OBB 的位置
        obb_frame.rotate(obb.R, center=(0, 0, 0))
        obb_frame.translate(obb.center)

        # 可视化
        o3d.visualization.draw_geometries([rack_pcd, obb, obb_frame])

        return self.rack_normal, self.rack_center

    def project_points_to_plane(self, points: np.ndarray) -> np.ndarray:
        """
        将点云投影到试管架平面

        计算每个点到平面的有向距离,然后将点投影到平面上
        Args:
            points (np.ndarray): 要投影的点云数据 (NumPy 数组)
        Returns:
            np.ndarray: 投影后的点云数据 (NumPy 数组)
        """
        # 计算每个点到平面的有向距离
        distances = np.dot(points - self.rack_center, self.rack_normal)

        # 投影点 = 原点 - 距离 * 法向量
        projected_points = points - distances[:, np.newaxis] * self.rack_normal

        return projected_points

    def compute_2d_convex_hull(self, projected_points: np.ndarray) -> np.ndarray:
        """
        计算投影点的 2D 凸包.

        将3D投影点转换为2D平面坐标,然后计算凸包.

        Args:
            projected_points (np.ndarray): 投影后的点云数据 (NumPy 数组).

        Returns:
            np.ndarray: 凸包的顶点坐标 (NumPy 数组).
        """
        # 将3D投影点转换为2D平面坐标
        points_2d = np.column_stack([
            np.dot(projected_points, self.rack_u_axis),
            np.dot(projected_points, self.rack_v_axis)
        ])

        if len(points_2d) < 3:
            return points_2d

        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            return hull_points
        except:
            return points_2d

    def find_tube_grid_position(self, tube_projected_center: np.ndarray) -> Tuple[int, int]:
        """
        根据试管投影中心找到对应的网格位置

        计算投影中心相对于试管架中心的位置偏移,然后转换为网格坐标

        Args:
            tube_projected_center (np.ndarray): 试管投影中心的坐标 (NumPy 数组)
        Returns:
            Tuple[int, int]: 试管在试管架上的网格位置 (行, 列)
        """
        # 将投影中心转换为2D坐标
        center_2d = np.array([
            np.dot(tube_projected_center, self.rack_u_axis),
            np.dot(tube_projected_center, self.rack_v_axis)
        ])

        # 计算试管架中心的2D坐标
        rack_center_2d = np.array([
            np.dot(self.rack_center, self.rack_u_axis),
            np.dot(self.rack_center, self.rack_v_axis)
        ])

        # 计算中心孔洞的平均位置
        center_row = sum(h[0] for h in self.CENTER_HOLES) / len(self.CENTER_HOLES)
        center_col = sum(h[1] for h in self.CENTER_HOLES) / len(self.CENTER_HOLES)

        # 计算相对偏移
        offset_2d = center_2d - rack_center_2d

        # 转换为网格坐标
        col_offset = offset_2d[0] / self.TUBE_SPACING
        row_offset = offset_2d[1] / self.TUBE_SPACING

        # 计算网格位置
        col = int(round(center_col + col_offset))
        row = int(round(center_row + row_offset))

        # 确保在有效范围内
        row = max(0, min(self.RACK_SHAPE[0] - 1, row))
        col = max(0, min(self.RACK_SHAPE[1] - 1, col))

        return (row, col)

    def calculate_overlap_score(self, hull_2d: np.ndarray, projected_center_2d: np.ndarray) -> float:
        """计算重合度得分"""
        if len(hull_2d) < 3:
            return 0.0

        try:
            # 创建凸包多边形
            hull_polygon = Polygon(hull_2d)

            # 创建理论试管圆形
            radius = self.TUBE_DIAMETER / 2.0
            circle = Point(projected_center_2d).buffer(radius)

            # 计算重叠面积
            intersection = hull_polygon.intersection(circle)
            union = hull_polygon.union(circle)

            if union.area == 0:
                return 0.0

            overlap_score = intersection.area / union.area
            return min(overlap_score, 1.0)
        except:
            return 0.0

    def process_tube_projection(self, tube_pcd: o3d.geometry.PointCloud,
                                tube_name: str, label: str = None,
                                confidence: float = None) -> TubeProjectionResult:
        """处理试管投影并找到网格位置"""
        print(f"处理试管投影: {tube_name}")

        # 获取试管点云
        tube_points = np.asarray(tube_pcd.points)

        if tube_points.size == 0:
            print(f"警告：{tube_name} 点云为空,跳过")
            return None

        tube_points_raw_center = np.mean(tube_points, axis=0)

        # 投影到试管架平面
        projected_points = self.project_points_to_plane(tube_points)
        # print("projected_points", projected_points)

        # 计算投影中心
        projected_center = np.mean(projected_points, axis=0)

        # 计算2D凸包
        hull_2d = self.compute_2d_convex_hull(projected_points)

        # 计算面积
        if len(hull_2d) >= 3:
            hull_polygon = Polygon(hull_2d)
            area = hull_polygon.area
        else:
            area = 0.0

        # 找到网格位置
        grid_position = self.find_tube_grid_position(projected_center)

        # 计算投影中心的2D坐标
        projected_center_2d = np.array([np.dot(projected_center, self.rack_u_axis),
                                        np.dot(projected_center, self.rack_v_axis)])

        # 计算重合度
        overlap_score = self.calculate_overlap_score(hull_2d, projected_center_2d)

        result = TubeProjectionResult(tube_name=tube_name,
                                      projected_center=projected_center,
                                      convex_hull=hull_2d,
                                      area=area,
                                      grid_position=grid_position,
                                      overlap_score=overlap_score,
                                      label=label,
                                      confidence=confidence)
        # 新增属性：原始3D中心
        result.tube_points_raw_center = tube_points_raw_center

        print(f"  标签: {label}, 置信度: {confidence:.3f}")
        print(f"  原始3D中心: {tube_points_raw_center}")
        print(f"  投影中心: {projected_center}")
        print(f"  网格位置: {grid_position}")
        print(f"  投影面积: {area:.6f}")
        print(f"  重合度得分: {overlap_score:.3f}")

        return result

    def visualize_multiple_tubes(self,
                                 tube_results: List[TubeProjectionResult],
                                 label_colors: dict,
                                 save_path: str = None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 1. 投影凸包
        ax1.set_title("所有试管投影凸包", fontsize=14)
        ax1.set_aspect('equal')

        # 用于图例去重
        legend_labels = set()

        for i, tube_result in enumerate(tube_results):
            # 根据label取颜色
            color = label_colors.get(tube_result.label, [0.5, 0.5, 0.5])

            # 绘制凸包
            if len(tube_result.convex_hull) > 2:
                hull_polygon = plt.Polygon(tube_result.convex_hull,
                                           fill=False, edgecolor=color, linewidth=2,
                                           label=tube_result.label if tube_result.label not in legend_labels else None)
                ax1.add_patch(hull_polygon)
                # 记录图例标签
                legend_labels.add(tube_result.label)

            # 绘制理论试管圆形
            center_2d = np.array([np.dot(tube_result.projected_center, self.rack_u_axis),
                                  np.dot(tube_result.projected_center, self.rack_v_axis)
                                  ])
            circle = plt.Circle(center_2d, self.TUBE_DIAMETER / 2,
                                fill=False, edgecolor=color, linewidth=2,
                                linestyle='--', alpha=0.7)

            ax1.add_patch(circle)

            # 标记中心点
            ax1.plot(center_2d[0], center_2d[1], 'o', color=color, markersize=8)
            ax1.text(center_2d[0], center_2d[1] + 0.005,
                     f'{tube_result.tube_name}',
                     ha='center', va='bottom', fontsize=8, color=color)

        ax1.grid(True, alpha=0.3)
        # 生成图例（自动去重后）
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   title="Tube Labels",
                   fontsize=8, title_fontsize=9)

        # ======================
        # 2. 试管架 (ax2)
        # ======================
        ax2.set_title("试管在试管架中的位置", fontsize=14)
        ax2.set_aspect('equal')

        legend_labels_ax2 = set()  # 图2的去重集合

        grid_map = {}
        for tube_result in tube_results:
            grid_map.setdefault(tube_result.grid_position, []).append(tube_result)

        for row in range(self.RACK_SHAPE[0]):
            for col in range(self.RACK_SHAPE[1]):
                x = col * self.TUBE_SPACING
                y = row * self.TUBE_SPACING

                if (row, col) in grid_map:
                    tubes_in_pos = grid_map[(row, col)]
                    if len(tubes_in_pos) == 1:
                        tube = tubes_in_pos[0]
                        color = label_colors.get(tube.label, [0.5, 0.5, 0.5])
                        circle = plt.Circle((x, y), self.TUBE_DIAMETER / 2,
                                            fill=True, facecolor=color, alpha=1,
                                            label=tube.label if tube.label not in legend_labels_ax2 else None)
                        ax2.add_patch(circle)
                        legend_labels_ax2.add(tube.label)  # 记录已添加 legend 的 label
                        ax2.text(x, y, f'({row},{col})',
                                 ha='center', va='center',
                                 fontsize=12, fontweight='bold', color='white')
                    else:
                        circle = plt.Circle((x, y), self.TUBE_DIAMETER / 2,
                                            fill=True, facecolor='red', alpha=1,
                                            label='冲突' if '冲突' not in legend_labels_ax2 else None)
                        ax2.add_patch(circle)
                        legend_labels_ax2.add('冲突')
                        ax2.text(x, y, f'冲突\n{len(tubes_in_pos)}个',
                                 ha='center', va='center',
                                 fontsize=8, fontweight='bold', color='white')
                else:
                    circle = plt.Circle((x, y), self.TUBE_DIAMETER / 2,
                                        fill=False, edgecolor='gray', alpha=0.5,
                                        label='空位' if '空位' not in legend_labels_ax2 else None)
                    ax2.add_patch(circle)
                    legend_labels_ax2.add('空位')

        ax2.set_xlim(-self.TUBE_SPACING, self.RACK_SHAPE[1] * self.TUBE_SPACING)
        ax2.set_ylim(-self.TUBE_SPACING, self.RACK_SHAPE[0] * self.TUBE_SPACING)
        ax2.grid(True, alpha=0.3)

        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   title="Tube Labels",
                   fontsize=8, title_fontsize=9)

        # ======================
        # 3. 表格
        # ======================
        table_data = []
        cell_colors = []
        for tube_result in tube_results:
            table_data.append([
                tube_result.tube_name,
                tube_result.label,
                f"{tube_result.confidence:.2f}",
                f"({tube_result.grid_position[0]}, {tube_result.grid_position[1]})",
                f"{tube_result.area:.6f}",
                f"{tube_result.overlap_score:.3f}"
            ])
            row_colors = ['white'] * 6
            row_colors[1] = label_colors.get(tube_result.label, [0.8, 0.8, 0.8])
            cell_colors.append(row_colors)

        col_labels = ["Name", "Label", "Conf", "Grid Pos", "Area", "Overlap"]
        table = ax2.table(cellText=table_data,
                          colLabels=col_labels,
                          cellColours=cell_colors,
                          loc='bottom', bbox=[0.0, -0.35, 1, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        for (_, _), cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")

        plt.show()


def create_detection_based_environment(robot_poses_data, base, obstacle_list=None):
    """
    基于检测结果创建仿真环境
    :param robot_poses_data: 检测结果数据
    :param base: Panda3D世界
    :return: (tubes_dict, obstacle_list, tube_rack_matrix, rack_shape)
    """

    # 3. 创建试管字典和障碍物列表
    tubes_dict = {}  # 使用字典存储,键为(row, col)
    # obstacle_list = []

    # 1. 加载试管架
    rack_info = robot_poses_data["rack_info"]
    rack_pose_homo = np.array(rack_info["pose_in_robot"], dtype=np.float64)
    rack_obj = cm.CollisionModel(OBJECT_MODEL_RACK)
    rack_obj.set_rgba([1.0, 0.0, 0.0, 1])
    rack_obj.set_pos(rack_pose_homo[:3, 3])
    rack_obj.set_rotmat(rack_pose_homo[:3, :3])
    rack_obj.attach_to(base)
    # obstacle_list.append(rack_obj)
    print("已添加试管架！")
    logging.info("已添加试管架到仿真环境")

    # 2. 获取检测矩阵信息
    rack_matrix = np.array(rack_info["rack_matrix"])
    print(f"检测矩阵:\n{rack_matrix}")
    logging.info(f"检测矩阵:\n{rack_matrix}")

    # 4. 根据检测结果加载试管
    detected_tubes = robot_poses_data["tubes_info"]

    for tube_info in detected_tubes:
        row, col = tube_info["grid_position"]

        # 使用试管位姿（不是帽子位姿）
        tube_pose = np.array(tube_info["Tube_center_pose_in_robot"], dtype=np.float64)

        # 读取 label（类别）
        label = tube_info.get("label", "default")
        config = TUBE_TYPE_CONFIG.get(label, TUBE_TYPE_CONFIG["default"])

        # 创建试管对象
        tube_obj = cm.CollisionModel(config["model_path"])
        tube_obj.set_rgba(config["color"])
        tube_obj.set_pos(tube_pose[:3, 3])
        tube_obj.set_rotmat(tube_pose[:3, :3])
        tube_obj.attach_to(base)

        tube_label = tube_info.get("label", "TubeB")  # 默认 TubeB
        tube_obj.tube_type = tube_label
        print(f"[DEBUG] 位置 ({row}, {col}) tube_type = {tube_obj.tube_type}")

        # 存储试管对象
        tubes_dict[(row, col)] = tube_obj
        obstacle_list.append(tube_obj)

        print(f"已添加试管位置 ({row}, {col}): {tube_info['tube_name']}, 置信度: {tube_info['confidence']:.3f}")
        logging.info(f"已添加试管位置 ({row}, {col}): {tube_info['tube_name']}, 置信度: {tube_info['confidence']:.3f}")

    # 5. 创建扔试管的目标位置（可以根据需要调整）
    # goal_pos = np.array([0.40, 0.30, 0.30])

    goal_pos = np.array([.40, .40, 0.25])
    goal_rotmat = np.array([[1.0, 0, 0],
                            [0, -1.0, 0],
                            [0, 0, -1.0]])

    print(f"共加载了 {len(tubes_dict)} 个试管")
    logging.info(f"共加载了 {len(tubes_dict)} 个试管")

    return tubes_dict, obstacle_list, rack_matrix, goal_pos, goal_rotmat


def print_detected_objects(detection_data):
    print("\n=== 检测到的物体列表 ===")
    for idx, obj in enumerate(detection_data):
        pos = np.round(obj["robot_xyz_m"], 3)
        print(f"{idx}: {obj['class_name']:<10} 位置: {pos}")


def go_init(obstacles):
    global rbt_r, rrtc_s
    init_jnts_values = np.array([0, 0, 0, 0, 0, 0])
    current_jnts = rbt_r.get_jnt_values()
    print("当前关节位置:", current_jnts)
    init = rrtc_s.plan(component_name="arm",
                       start_conf=current_jnts,
                       goal_conf=init_jnts_values,
                       obstacle_list=obstacles,
                       ext_dist=0.01,
                       max_time=300)
    if init is None:
        raise RuntimeError("初始化路径规划失败")
    rbt_r.move_jntspace_path(init, speed_n=100)


# config_camera.py
CAMERA_INTRINSICS = {
    "fx": 606.627,
    "fy": 606.657,
    "cx": 324.281,
    "cy": 241.149
}

CAMERA_INTRINSICS_NEW = {
    "fx": 605.492,
    "fy": 604.954,
    "cx": 326.026,
    "cy": 243.012
}
