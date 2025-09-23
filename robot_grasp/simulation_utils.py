#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：ABB_wrs_hu_sam
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
log_filename = f'grasping_log_detection_{timestamp}.txt'

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


def save_yolo_detection_image(img, detections, save_path, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 YOLO 检测到的边界框和标签 ,并保存图像

    Args:
        img (numpy.ndarray): 原始图像
        detections (list): YOLOv9 检测结果列表 (xyxy 格式)
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
                          target_label: str,
                          check_rotate: bool = False
                          ) -> Tuple[
    Optional[np.ndarray], bool, Optional[Tuple[float, float, float, float]], Optional[float], str]:
    """
    从 YOLOv9 检测结果中选择面积最大的目标对象

    Args:
        detections_tensor: YOLOv9 检测结果张量列表
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
            current_label = class_names.get(int(label), "Unknown")

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

    def __init__(self, rack_shape=(5, 10), center_holes=[(2, 4), (2, 5)],
                 tube_spacing=0.025, tube_diameter=0.021):
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
            radius = TUBE_DIAMETER / 2.0
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
                    if len(tubes_in_pos) != 1:
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


class Rack510Detector:
    def __init__(self, color_path, depth, weights_path, T_tcp_in_robot, save_dirs):
        """
        Rack510 检测与位姿估计类
        :param weights_path: YOLO 模型路径
        :param T_tcp_in_robot: TCP在机器人坐标系下的位姿
        """
        # === 保存配置 ===
        self.weights_path = weights_path
        self.T_tcp_in_robot = T_tcp_in_robot
        self.save_dirs = save_dirs

        # === 数据保存路径 ===
        self.SAVE_HRC_PATHS_DIR = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\HRC_paths"
        os.makedirs(self.SAVE_HRC_PATHS_DIR, exist_ok=True)

        # === 模板点云路径 ===
        base_obj_dir = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects"
        self.template_cloud_rack = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack510_plane.ply"))
        self.template_cloud_rack_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack510.ply"))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_rack_pose_in_robot(self, T_tcp_in_robot,
                                   T_cam_in_flange=None,
                                   T_tcp_offset=None,
                                   T_rack_in_cam=None):

        """
        计算试管架在机器人坐标系下的位姿
        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_rack_in_cam: (4x4) 试管架在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) rack_pose_robot
        """
        # 默认 T_cam_in_flange（标定结果）
        if T_cam_in_flange is None:
            T_cam_in_flange = np.array([[-0.0092273, -0.99984, -0.01527, 0.092345],
                                        [0.99995, -0.0092737, 0.0029707, -0.036271],
                                        [-0.0031118, -0.015242, 0.99988, 0.032591],
                                        [0, 0, 0, 1]
                                        ])

        # 默认 TCP -> 法兰 位移
        if T_tcp_offset is None:
            T_tcp_offset = np.array([0, 0, 0.21285675])  # 单位 m

        # 已知 TCP 相对于法兰的位姿 (固定)
        T_tcp_in_flange = np.eye(4)
        T_tcp_in_flange[:3, :3] = np.eye(3)
        T_tcp_in_flange[:3, 3] = T_tcp_offset

        # flange 相对于 robot 的位姿
        T_flange_in_robot = T_tcp_in_robot @ np.linalg.inv(T_tcp_in_flange)
        print("[DEBUG] T_flange_in_robot:\n", T_flange_in_robot)

        if T_rack_in_cam is None:
            raise ValueError("必须传入 T_rack_in_cam (试管架在相机坐标系下的位姿)")

        # 最终 rack_pose_robot
        rack_pose_robot = T_flange_in_robot @ T_cam_in_flange @ T_rack_in_cam
        print("[DEBUG] rack_pose_robot:\n", rack_pose_robot)

        return rack_pose_robot

    def process_rack510_from_image(self, color_img, depth_img, pred, mask_dir, pointcloud_dir):
        """
        从 RGB + 深度图像中检测并配准 rack510,返回 rack_pose_cam
        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: rack_pose_cam (4x4 numpy array) 或 None
        """
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # ---- 1. 找 rack510 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results, "rack510",
                                                                                         check_rotate=True)
        if largest_rack is None:
            print("[ERROR] 未检测到 rack510")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx, fy, cx, cy = CAMERA_INTRINSICS.values()
        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(
            point_cloud_path,
            os.path.join(self.save_dirs["plane"]),
            os.path.join(self.save_dirs["remaining"]),
            0.002, 20000
        )

        # ---- 5. ICP 位姿估计 ----
        rack_pose_cam = icp_plane_estimate_pose_rack_tube(label_name, plane_cloud, self.template_cloud_rack)
        print("试管架在相机坐标系下的位姿：\n", rack_pose_cam)

        # ---- 6. 可视化对比 ----
        template_cloud_rack_full = copy.deepcopy(self.template_cloud_rack_full)
        aligned_scene_pcd = template_cloud_rack_full.transform(rack_pose_cam)

        # 给颜色区分
        point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
        aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

        o3d.visualization.draw_geometries(
            [aligned_scene_pcd, point_cloud],
            window_name="After ICP Registration",
            width=1024,
            height=768
        )

        if rack_pose_cam is not None:
            # 试管架在相机坐标系下的位姿
            rack_pose_robot = self.compute_rack_pose_in_robot(T_tcp_in_robot=T_tcp_in_robot,
                                                              T_rack_in_cam=rack_pose_cam)

            print("[DEBUG] rack_pose_robot:\n", rack_pose_robot)

        return rack_pose_cam, pred_results, template_cloud_rack


class TubeProcessor:
    def __init__(self, rack_pose_cam, pred_results, template_cloud_rack, rack_shape=(5, 10)):
        """
        :param rack_shape: 试管架形状 (行, 列)
        """
        self.RACK_SHAPE = rack_shape
        self.LABEL_TO_NUM = {
            "TubeB": 1,
            "TubeG": 2,
            "TubeO": 3
        }
        self.label_colors = {
            'TubeB': [0, 0, 1],
            'TubeG': [0, 1, 0],
            'TubeO': [1.0, 0.65, 0.0],
        }

        self.TUBE_SPACING = 0.025  # m
        self.CENTER_HOLES = [(2, 4), (2, 5)]  # 中心空孔位置
        self.TUBE_DIAMETER = 0.021  # m
        self.rack_pose_cam = rack_pose_cam,
        self.pred_results = pred_results,
        self.template_cloud_rack = template_cloud_rack

    def process_tube(self, color_img, depth_image, pred_results,
                     plane_output_folder, remaining_output_folder,
                     mask_output_folder, point_cloud_output_folder,
                     rack_pose_cam, rack_pose_robot,
                     save_folder):
        """
        完整试管架 + 试管处理流水线
        """
        fx, fy, cx, cy = CAMERA_INTRINSICS.values()
        tube_data_list = extract_tube_point_clouds(color_img, depth_image, pred_results, ['TubeB', 'TubeG', 'TubeO'],
                                                   fx, fy, cx, cy, point_cloud_output_folder,
                                                   mask_output_folder, plane_output_folder, remaining_output_folder)

        rack_pcd = template_cloud_rack
        print(f"试管架点云点数: {len(rack_pcd.points)}")

        # 变换矩阵
        transformation_matrix = rack_pose_cam
        # 应用变换矩阵
        rack_pcd.transform(transformation_matrix)

        # 创建估计器
        estimator = SimplifiedTubePoseEstimator()

        # 计算试管架平面信息
        normal, center = estimator.compute_rack_plane_info(rack_pcd)

        # 处理所有试管
        print("\n处理所有试管...")
        tube_results = []
        for tube_data in tube_data_list:
            tube_id = tube_data["tube_id"]
            tube_pcd = tube_data["pcd"]
            label = tube_data["label"]
            confidence = tube_data["confidence"]

            result = estimator.process_tube_projection(tube_pcd, tube_id, label=label, confidence=confidence)
            if result:
                tube_results.append(result)

        # -------------------------
        # 点云上色（保证颜色和标签一致）
        # -------------------------
        print("\n可视化所有点云...")
        rack_pcd.paint_uniform_color([1.0, 0, 0.0])
        label_colors = {
            'TubeB': [0, 0, 1],
            'TubeG': [0, 1, 0],
            'TubeO': [1.0, 0.65, 0.0],
        }

        for tube_data in tube_data_list:
            color = label_colors.get(tube_data["label"], [0.5, 0.5, 0.5])  # 默认灰色
            tube_data["pcd"].paint_uniform_color(color)

        # -------------------------
        # 可视化
        # -------------------------
        geometries = [rack_pcd] + [tube_data["pcd"] for tube_data in tube_data_list]
        o3d.visualization.draw_geometries(geometries,
                                          window_name="所有试管点云",
                                          width=1024,
                                          height=768)

        # -------------------------
        # 输出位置冲突检查
        # -------------------------
        print("\n位置冲突检查...")
        position_map = {}
        for result in tube_results:
            pos = result.grid_position
            position_map.setdefault(pos, []).append(result.tube_name)

        conflicts = [(pos, tubes) for pos, tubes in position_map.items() if len(tubes) > 1]
        if conflicts:
            for pos, tubes in conflicts:
                print(f"  位置 {pos}: {', '.join(tubes)}")
        else:
            print("  无位置冲突.")

        # 可视化结果
        estimator.visualize_multiple_tubes(tube_results, label_colors=label_colors,
                                           save_path="multiple_tubes_projection_result.png")

        # =============================================================================================

        # 试管位姿计算
        tubes_robot_info = []
        for result in tube_results:
            # 投影中心在相机坐标系下
            tube_center_cam = result.tube_points_raw_center  # [x, y, z] in camera coords

            # 齐次坐标
            tube_center_cam_h = np.append(tube_center_cam, 1.0)  # (4,)

            # 转换到机器人坐标系
            tube_center_robot_h = T_flange_in_robot @ T_cam_in_flange @ tube_center_cam_h
            tube_center_robot = tube_center_robot_h[:3]

            rack_rot_robot = np.array(rack_pose_robot)[:3, :3]  # 取旋转部分
            # 构建试管的 4x4 齐次变换矩阵
            tube_pose_robot = np.eye(4)
            tube_pose_robot[:3, :3] = rack_rot_robot  # 旋转用试管架的
            tube_pose_robot[:3, 3] = tube_center_robot  # 平移用试管自身中心

            tubes_robot_info.append({
                "tube_name": result.tube_name,
                "label": result.label,
                "confidence": float(result.confidence),
                "grid_position": result.grid_position,
                "Tube_center_pose_in_robot": tube_pose_robot.tolist(),
                "area": float(result.area),
                "overlap_score": float(result.overlap_score)
            })

        # ------------------------------
        # 生成试管架状态矩阵
        # ------------------------------

        rack_matrix = [[0 for _ in range(RACK_SHAPE[1])] for _ in range(RACK_SHAPE[0])]

        for result in tube_results:
            row, col = result.grid_position
            num_label = self.LABEL_TO_NUM.get(result.label, 0)  # 如果找不到对应的试管标签则设为 0
            rack_matrix[row][col] = num_label

        # 保存 JSON
        robot_poses_data = {
            "rack_info": {
                "rack_label": rack_label_name,
                "rack_confidence": float(rack_confidence),
                "pose_in_camera": T_rack_in_cam.tolist(),
                "pose_in_robot": rack_pose_robot.tolist(),
                "rack_matrix": rack_matrix
            },
            "tubes_info": tubes_robot_info
        }

        save_json_path = os.path.join(save_folder, "rack_and_tubes_pose_robot.json")
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)

        # 打印成目标格式
        print("rack_matrix")

        def print_matrix_format(matrix):
            print("[")
            for row in matrix:
                print(f"    {row},")
            print("]")

        print_matrix_format(rack_matrix)
        print(f"试管架和试管的位姿信息已保存到: {save_json_path}")


if __name__ == "__main__":
    import modeling.collision_model as cm
    import copy
    import json
    import math
    import os
    from time import sleep
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_con.gofa_con.gofa_con as gofa_con
    import robot_sim.robots.gofa5.gofa5_Ag145 as gf5
    import motion.probabilistic.rrt_connect as rrtc
    import drivers.devices.dh.maingripper as dh_r
    import robot_sim.end_effectors.gripper.ag145.ag145 as dh

    # ========================= 配置区域 =========================
    json_path = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\kinect_data_global\detection_results.json"

    object_model_map = {
        "ore": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\refrigerator_open.STL",
        "cre": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\refrigerator_yuan.STL",
        "oce": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\centrifuge_machine.STL",
        "cce": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\centrifuge_machine.STL",
        "locker": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\locker.STL",
        "rack510": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack510.STL",
        "rack34": r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack_10ml_center.STL"
    }

    object_color_map = {"ore": [1, 0, 0, 1],
                        "cre": [1, 0, 0, 1],
                        "oce": [0.5, 0.5, 0.5, 1],
                        "cce": [0.5, 0.5, 0.5, 1],
                        "locker": [1, 0.5, 0, 1],
                        "rack510": [1, 1, 1, 1],
                        "rack34": [0.5, 0.8, 1, 1],
                        }

    # Step 1: 建立仿真环境
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    # gm.gen_frame().attach_to(base)

    # Step 2: 创建机器人对象
    robot_s = gf5.GOFA5()
    rbt_r = gofa_con.GoFaArmController()

    grip_r = dh_r.MainGripper('com3')
    gripper_s = dh.Ag145()
    robot_s.hnd.jaw_to(0.023 / 2)
    grip_r.jaw_to(0.023 / 2)

    rrtc_s = rrtc.RRTConnect(robot_s)
    # robot_s.gen_meshmodel().attach_to(base)
    move_real_rbt = True

    # 加载检测结果
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] {json_path} 不存在！")
    with open(json_path, "r", encoding="utf-8") as f:
        detection_data = json.load(f)

    obstacles = []
    rotmat_stl = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
    for obj in detection_data:
        class_name = obj["class_name"]
        if class_name in object_model_map and os.path.exists(object_model_map[class_name]):
            model_path = object_model_map[class_name]
            obj_model = cm.CollisionModel(model_path)
            obj_model.set_pos(obj["robot_xyz_m"])
            obj_model.set_rotmat(rotmat_stl)
            color = object_color_map.get(class_name, [1, 1, 1, 0.8])
            obj_model.set_rgba(color)
            # obj_model.attach_to(base)
            # 只添加非 "rack510" 的物体到障碍物列表
            if class_name != "rack510":
                obstacles.append(obj_model)
            else:
                # 保存 rack510 的位置
                rack510_pos = obj["robot_xyz_m"]
            # obstacles.append(obj_model)
            obj_model.show_cdprimit()

    print_detected_objects(detection_data)

    # 如果控制真实机器人,提前连接
    if move_real_rbt:
        go_init(obstacles)
        print("机械臂已初始化完成")
    # # base.run()

    # pos_1 = np.array([0.7, 0, 0.3])

    # Step 3: 目标位姿 & IK
    if rack510_pos is not None:
        pos_1 = rack510_pos + np.array([-0.1, 0, 0.2])
    else:
        print("[!] 未检测到 rack510,使用默认位置")
        pos_1 = np.array([0.7, 0, 0.3])  # 默认位置
    print("检测到 rack510,使用检测到的位置", pos_1)
    rot_1 = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
    print("目标位姿：", pos_1, rot_1)

    app1_jnt_values = robot_s.ik(tgt_pos=pos_1, tgt_rotmat=rot_1)
    if app1_jnt_values is None:
        raise ValueError(f"IK failed for pos={pos_1}, rot={rot_1}")
    print("app_1的关节坐标:", app1_jnt_values)

    robot_s.fk(component_name='arm', jnt_values=app1_jnt_values)

    # ----------------------------
    # 新增：计算法兰盘（夹爪连接处）位姿
    flange_pos = robot_s.arm.jnts[-1]['gl_posq']
    flange_rot = robot_s.arm.jnts[-1]['gl_rotmatq']
    print("法兰盘位置:", flange_pos)
    print("法兰盘姿态:\n", flange_rot)

    # 获取夹爪当前中心（随开度变化）
    current_jaw_center = robot_s.hnd.jaw_center_pos  # 这里会反映当前 jaw_to() 的开度

    # 计算 TCP（夹爪中心全局坐标）
    tcp_pos = flange_pos + flange_rot @ current_jaw_center
    tcp_rot = flange_rot  # 方向和法兰一致

    print("动态计算的 TCP 位置:", tcp_pos)
    print("动态计算的 TCP 姿态:\n", tcp_rot)
    # ----------------------------

    robot_s.gen_meshmodel().attach_to(base)

    # Step 4: 路径规划
    # init_jnt_values = np.zeros(6)

    current_conf = robot_s.get_jnt_values("arm")  # 获取当前真实关节值

    point_1 = None
    for i in range(100):
        point_1 = rrtc_s.plan(component_name="arm",
                              start_conf=current_conf,
                              goal_conf=app1_jnt_values,
                              obstacle_list=obstacles,
                              ext_dist=0.01,
                              max_time=100)
        if point_1 is not None and len(point_1) > 0:
            print(f"[✓] 第 {i + 1} 次规划成功")
            break
        else:
            print(f"[×] 第 {i + 1} 次规划失败,重试...")

    if point_1 is None or len(point_1) == 0:
        print("[!] 最终规划失败")

    # 先进行仿真检测
    # base.run()

    # Step 5: 执行
    if move_real_rbt:
        rbt_r.move_jntspace_path(point_1)
        # rbt_r.move_jntspace_path(path_end)
    else:
        robot_s.gen_meshmodel().attach_to(base)

    # # Step 6: 最后一次运行可视化
    # base.run()

    # 配置参数
    RACK_SHAPE = (5, 10)  # 试管架形状 (行,列)
    TUBE_SPACING = 0.025  # 试管间距 (m)
    CENTER_HOLES = [(2, 4), (2, 5)]  # 中心孔洞位置
    TUBE_DIAMETER = 0.021  # 试管直径 (m)

    # 相机内参
    fx = 606.627
    fy = 606.657
    cx = 324.281
    cy = 241.149

    # # 新相机
    # fx = 605.492
    # fy = 604.954
    # cx = 326.026
    # cy = 243.012

    save_HRC_paths_dir = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\HRC_paths"
    template_cloud_rack = o3d.io.read_point_cloud(
        r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack510_plane.ply")
    template_cloud_rack_quan = o3d.io.read_point_cloud(
        r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack510.ply")

    # 清空文件夹
    clear_folder(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(point_cloud_output_folder, exist_ok=True)
    os.makedirs(save_dir_pcd, exist_ok=True)

    # 初始化 YOLOv9 API
    weights = 'best_rack510.pt'
    csv_path = os.path.join(save_dir_pcd, 'detection_results.csv')
    detect_api = DetectAPI(weights=weights, csv_path=csv_path)

    # 2. Intel Realsense D435 相机的点云获取与预处理
    color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
        save_dir_pcd)

    # 读取图像
    source = save_colorpath
    depth_image_path = save_depthpath

    img = cv2.imread(source)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)  # 读取深度图像
    img_list = [img]  # YOLOv9 API 接受图像列表

    # YOLOv9 检测
    img_detected, pred, total_objects = detect_api.detect(img_list)  # 使用 YOLOv9 API 进行检测
    # print("pred", pred)

    # 保存 img_detected
    img_detected_path = os.path.join(save_dir_pcd, 'img_detected.jpg')
    cv2.imwrite(img_detected_path, img_detected)
    print(f"img_detected 已保存: {img_detected_path}")

    # 4. 选择面积最大的试管架,并判断是否需要旋转矩阵
    largest_rack, rotate_matrix, bbox_rack, rack_confidence, rack_label_name = select_largest_object(pred,
                                                                                                     target_label="rack510",
                                                                                                     check_rotate=True)
    print(f"检测到目标: {rack_label_name}, 置信度: {rack_confidence:.2f}, 边界框: {bbox_rack}")

    # SAM 分割
    mask = sam_segment(img, bbox_rack)

    # 保存目标掩码图像 mask_output_folder
    mask_image, mask_path = save_object_mask_on_black_background(img, mask, bbox_rack, rack_label_name,
                                                                 mask_output_folder)
    mask = cv2.imread(mask_path)
    point_cloud = extract_point_cloud(img, mask_image, depth_image, fx, fy, cx, cy)

    point_cloud_path = os.path.join(point_cloud_output_folder, f"point_cloud_{rack_label_name}.ply")
    o3d.io.write_point_cloud(point_cloud_path, point_cloud)
    print(f"物体 {rack_label_name} 的点云已保存: {point_cloud_path}")

    # 提取平面
    plane_cloud, remaining_cloud, plane_filename, remaining_filename = extract_plane(point_cloud_path,
                                                                                     plane_output_folder,
                                                                                     remaining_output_folder,
                                                                                     threshold=0.002,
                                                                                     max_iterations=20000)

    print(f"物体 {rack_label_name} 的平面点云已保存: {plane_filename}")
    print(f"物体 {rack_label_name} 的剩余点云已保存: {remaining_filename}")

    # 估计位姿
    rack_pose = icp_plane_estimate_pose_rack_tube(rack_label_name, plane_cloud, template_cloud_rack)
    print("试管架的位姿：", rack_pose)

    template_cloud_quan = copy.deepcopy(template_cloud_rack_quan)
    aligned_scene_pcd = template_cloud_quan.transform(rack_pose)

    point_cloud.paint_uniform_color([1, 0, 0])
    aligned_scene_pcd.paint_uniform_color([0, 1, 0])

    # 5. 可视化对比
    o3d.visualization.draw_geometries([aligned_scene_pcd, point_cloud],
                                      window_name="After ICP_Iterative_Closest_Point Registration",
                                      width=1024, height=768)

    # 试管架在相机坐标系下的位姿
    T_rack_in_cam = rack_pose

    # =============================    计算 T_cam_in_flange 的参数信息    ===========================

    # # 试管架在机器人坐标系下的位姿
    # T_rack_in_robot = np.array([[0, - 1, 0, 0.8],
    #                             [-1, 0, 0, 0],
    #                             [0, 0, -1, 0.095 - 0.015],
    #                             [0, 0, 0, 1]])

    # T_rack_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_rack_in_cam

    # T_tcp_in_robot = np.array([
    #     [1, 0, 0, 0.7],
    #     [0, -1, 0, 0.0],
    #     [0, 0, -1, 0.3],
    #     [0, 0, 0, 1.0]
    # ])

    T_tcp_in_robot = np.eye(4)  # 创建一个 4x4 的单位矩阵
    T_tcp_in_robot[:3, :3] = rot_1  # 将旋转矩阵 rot_1 放入位姿矩阵的左上角 3x3 部分
    T_tcp_in_robot[:3, 3] = pos_1  # 将位置向量 pos_1 放入位姿矩阵的右上角 3x1 部分
    print("T_tcp_in_robot", T_tcp_in_robot)

    # 已知 TCP 相对于法兰的位姿 (固定)
    T_tcp_in_flange = np.eye(4)
    T_tcp_in_flange[:3, :3] = np.eye(3)
    T_tcp_in_flange[:3, 3] = np.array([0, 0, 0.21285675])  # 0.1823

    # 求 flange 相对于 robot 的位姿
    T_flange_in_robot = T_tcp_in_robot @ np.linalg.inv(T_tcp_in_flange)
    print("T_flange_in_robot", T_flange_in_robot)

    # # 求相机在夹爪坐标系下的位姿  T_cam_in_flange
    # T_cam_in_flange = np.linalg.inv(T_flange_in_robot) @ T_rack_in_robot @ np.linalg.inv(T_rack_in_cam)
    # print("T_cam_in_flange:\n", T_cam_in_flange)

    T_cam_in_flange = np.array([[-0.0092273, -0.99984, -0.01527, 0.092345],
                                [0.99995, -0.0092737, 0.0029707, -0.036271],
                                [-0.0031118, -0.015242, 0.99988, 0.032591],
                                [0, 0, 0, 1]])

    # T_cam_in_flange:
    #  [[  -0.037917    -0.99921   -0.012065    0.094168]
    #  [    0.99923   -0.038033   0.0095435   -0.035118]
    #  [ -0.0099948   -0.011694     0.99988    0.032959]
    #  [          0           0           0           1]]

    # ------------------------------------------
    # 计算试管架和试管在机器人坐标系下的位姿并保存 JSON
    # ------------------------------------------
    rack_pose_robot = T_flange_in_robot @ T_cam_in_flange @ T_rack_in_cam
    print("rack_pose_robot", rack_pose_robot)

    # # 固定高度表
    # object_distance = {
    #     "ore": 0.35 - 0.015,  # 0.335
    #     "cre": 0.35 - 0.015,  # 0.335
    #     "oce": 0.21 - 0.015,  # 0.195
    #     "cce": 0.21 - 0.015,  # 0.195
    #     "locker": 0.31 - 0.015,  # 0.3
    #     "rack510": 0.95 - 0.015,  # 0.08
    #     "rack34": 0.12 - 0.015,  # 0.105
    # }
    #
    # # ======= 修改 Z 坐标为实际高度 =======
    # if class_name in object_distance:
    #     robot_xyz[2] = object_distance[class_name]
    #     rack_pose_robot[2, 3] = object_distance[class_name]

    # ==========================================================================================

    print("--------------------------------    试管点云分割 与 点云投影计算   ---------------------------")
    # 分割试管帽点云
    tube_data_list = extract_tube_point_clouds(img, depth_image, pred, ['TubeB', 'TubeG', 'TubeO'],
                                               606.627, 606.657, 324.281, 241.149,
                                               point_cloud_output_folder,
                                               mask_output_folder, plane_output_folder, remaining_output_folder)

    rack_template_pcd_path = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\rack510_plane.ply"

    # 读取试管架点云
    print("\n读取试管架点云文件...")
    rack_pcd = o3d.io.read_point_cloud(rack_template_pcd_path)
    print(f"试管架点云点数: {len(rack_pcd.points)}")
    # 变换矩阵
    transformation_matrix = rack_pose
    # 应用变换矩阵
    rack_pcd.transform(transformation_matrix)

    # 创建估计器
    estimator = SimplifiedTubePoseEstimator()

    # 计算试管架平面信息
    normal, center = estimator.compute_rack_plane_info(rack_pcd)

    # 处理所有试管
    print("\n处理所有试管...")
    tube_results = []
    for tube_data in tube_data_list:
        tube_id = tube_data["tube_id"]
        tube_pcd = tube_data["pcd"]
        label = tube_data["label"]
        confidence = tube_data["confidence"]

        result = estimator.process_tube_projection(tube_pcd, tube_id, label=label, confidence=confidence)
        if result:
            tube_results.append(result)

    # -------------------------
    # 点云上色（保证颜色和标签一致）
    # -------------------------
    print("\n可视化所有点云...")
    rack_pcd.paint_uniform_color([1.0, 0, 0.0])
    label_colors = {
        'TubeB': [0, 0, 1],
        'TubeG': [0, 1, 0],
        'TubeO': [1.0, 0.65, 0.0],
    }

    for tube_data in tube_data_list:
        color = label_colors.get(tube_data["label"], [0.5, 0.5, 0.5])  # 默认灰色
        tube_data["pcd"].paint_uniform_color(color)

    # -------------------------
    # 可视化
    # -------------------------
    geometries = [rack_pcd] + [tube_data["pcd"] for tube_data in tube_data_list]
    o3d.visualization.draw_geometries(geometries,
                                      window_name="所有试管点云",
                                      width=1024,
                                      height=768)

    # -------------------------
    # 输出位置冲突检查
    # -------------------------
    print("\n位置冲突检查...")
    position_map = {}
    for result in tube_results:
        pos = result.grid_position
        position_map.setdefault(pos, []).append(result.tube_name)

    conflicts = [(pos, tubes) for pos, tubes in position_map.items() if len(tubes) > 1]
    if conflicts:
        for pos, tubes in conflicts:
            print(f"  位置 {pos}: {', '.join(tubes)}")
    else:
        print("  无位置冲突.")

    # 可视化结果
    estimator.visualize_multiple_tubes(tube_results, label_colors=label_colors,
                                       save_path="multiple_tubes_projection_result.png")

    # =============================================================================================

    # 试管位姿计算
    tubes_robot_info = []
    for result in tube_results:
        # 投影中心在相机坐标系下
        tube_center_cam = result.tube_points_raw_center  # [x, y, z] in camera coords

        # 齐次坐标
        tube_center_cam_h = np.append(tube_center_cam, 1.0)  # (4,)

        # 转换到机器人坐标系
        tube_center_robot_h = T_flange_in_robot @ T_cam_in_flange @ tube_center_cam_h
        tube_center_robot = tube_center_robot_h[:3]

        rack_rot_robot = np.array(rack_pose_robot)[:3, :3]  # 取旋转部分
        # 构建试管的 4x4 齐次变换矩阵
        tube_pose_robot = np.eye(4)
        tube_pose_robot[:3, :3] = rack_rot_robot  # 旋转用试管架的
        tube_pose_robot[:3, 3] = tube_center_robot  # 平移用试管自身中心

        tubes_robot_info.append({
            "tube_name": result.tube_name,
            "label": result.label,
            "confidence": float(result.confidence),
            "grid_position": result.grid_position,
            "Tube_center_pose_in_robot": tube_pose_robot.tolist(),
            "area": float(result.area),
            "overlap_score": float(result.overlap_score)
        })

    # ------------------------------
    # 生成试管架状态矩阵
    # ------------------------------
    # 定义标签映射关系
    LABEL_TO_NUM = {
        "TubeB": 1,
        "TubeG": 2,
        "TubeO": 3
    }

    rack_matrix = [[0 for _ in range(RACK_SHAPE[1])] for _ in range(RACK_SHAPE[0])]

    for result in tube_results:
        row, col = result.grid_position
        num_label = LABEL_TO_NUM.get(result.label, 0)  # 如果找不到对应的试管标签则设为 0
        rack_matrix[row][col] = num_label

    # 保存 JSON
    robot_poses_data = {
        "rack_info": {
            "rack_label": rack_label_name,
            "rack_confidence": float(rack_confidence),
            "pose_in_camera": T_rack_in_cam.tolist(),
            "pose_in_robot": rack_pose_robot.tolist(),
            "rack_matrix": rack_matrix
        },
        "tubes_info": tubes_robot_info
    }

    save_json_path = os.path.join(save_folder, "rack_and_tubes_pose_robot.json")
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)

    # 打印成目标格式
    print("rack_matrix")


    def print_matrix_format(matrix):
        print("[")
        for row in matrix:
            print(f"    {row},")
        print("]")


    print_matrix_format(rack_matrix)
    print(f"试管架和试管的位姿信息已保存到: {save_json_path}")

    ###################### 场景位姿信息检测完毕,开始人机协作机械臂抓取工作 ######################################
    print("场景位姿信息检测完毕,开始人机协作机械臂抓取工作......")
    print("=" * 30)

    # base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    # gm.gen_frame().attach_to(base)
    #
    # rbt_s = gf5.GOFA5()  # 创建 GOFA5 机器人对象
    # # rbt_s.gen_meshmodel().attach_to(base)  # 生成机器人网格模型并添加到世界
    # rbt_s.hnd.jaw_to(0.023 / 2)

    # # 读取检测结果
    # try:
    #     robot_poses_data = load_robot_poses(POSE_JSON_PATH)
    #     print("成功加载检测结果JSON文件")
    #     logging.info("成功加载检测结果JSON文件")
    # except Exception as e:
    #     print(f"加载JSON文件失败: {e}")
    #     logging.error(f"加载JSON文件失败: {e}")
    #     sys.exit(1)

    # 基于检测结果创建环境
    print("正在创建仿真环境...")
    tubes_dict, obstacle_list, tube_rack_matrix, goal_pos, goal_rotmat = create_detection_based_environment(
        robot_poses_data, base, obstacles)
    # base.run()

    # 初始化机器人抓取器
    robot_grasp = RobotGrasping(base, robot_s)

    print("当前夹爪宽度:", robot_s.get_gripper_width())

    # 创建保存路径的文件夹
    save_dir = os.path.join(save_HRC_paths_dir, "saved_paths_animation_detection")

    # 清空文件夹
    clear_folder(save_HRC_paths_dir)

    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 抓取位置信息列表
    grasp_positions = []
    grasp_count = 0

    # 主循环
    while np.sum(tube_rack_matrix) > 0:  # 只要还有试管就继续
        robot_positions = find_robot_grabbable_positions(tube_rack_matrix)  # 寻找机器人可抓取的位置

        if robot_positions:
            logging.info("机器人可以抓取的位置:")
            for pos in robot_positions:
                logging.info(f"  - ({pos[0]}, {pos[1]})")

            # 机器人可以抓取
            robot_row, robot_col = robot_positions[0]  # 简单地选择第一个可抓取位置

            # 检查该位置是否有试管对象
            if (robot_row, robot_col) not in tubes_dict:
                logging.warning(f"位置 ({robot_row}, {robot_col}) 没有对应的试管对象")
                print(f"位置 ({robot_row}, {robot_col}) 没有对应的试管对象")
                break

            tube_to_grasp = tubes_dict[(robot_row, robot_col)]
            print("tube_to_grasp", tube_to_grasp)

            # print("=== tube_to_grasp 对象分析 ===")
            #
            # # 基本信息
            # print(f"对象类型: {type(tube_to_grasp)}")
            # print(f"对象值: {tube_to_grasp}")
            #
            # # 所有属性和方法
            # all_attributes = dir(tube_to_grasp)
            # print(f"所有属性和方法 ({len(all_attributes)} 个):")
            # for attr in all_attributes:
            #     if not attr.startswith('_'):  # 过滤掉私有属性
            #         try:
            #             value = getattr(tube_to_grasp, attr)
            #             print(f"  {attr}: {value}")
            #         except Exception as e:
            #             print(f"  {attr}: <无法访问: {e}>")
            #
            #
            ## +++======

            # 机器人抓取
            logging.info(f"程序判断:机器人抓取位置:({robot_row}, {robot_col})")
            print(f"程序判断:机器人抓取位置:({robot_row}, {robot_col})")

            action = input("是否执行机器人抓取？ (y/n): ")

            # action = 'y'  # 自动化模式,直接执行抓取

            current_conf = robot_s.get_jnt_values("arm")  # 获取当前真实关节值

            # rbt_s.fk(component_name='arm', jnt_values=app1_jnt_values)

            if action.lower() == 'y':
                result = robot_grasp.grasp_tube_invocation(tube_to_grasp, goal_pos, goal_rotmat, obstacle_list,
                                                              robot_row, robot_col, current_conf, save_HRC_paths_dir)

                if result is not None and isinstance(result, tuple) and len(result) == 3:
                    conf_list, jawwidth_list, objpose_list = result

                    if conf_list and len(conf_list) > 0:
                        # 保存抓取路径信息
                        path_data = {
                            'conf_list': conf_list,
                            'jawwidth_list': jawwidth_list,
                            'objpose_list': objpose_list,
                            'row': robot_row,
                            'col': robot_col,
                            'detection_info': robot_poses_data["tubes_info"]
                        }

                        path_filename = os.path.join(save_dir, f"grasp_path_{robot_row}_{robot_col}.pkl")
                        with open(path_filename, 'wb') as f:
                            pickle.dump(path_data, f)
                        logging.info(f"成功保存抓取路径到: {path_filename}")
                        print(f"成功保存抓取路径到: {path_filename}")

                        # 保存抓取位置信息
                        grasp_positions.append({"row": robot_row, "col": robot_col, "type": "Robot"})

                        # 移除试管
                        robot_grasp.detach_tube(tube_to_grasp)
                        del tubes_dict[(robot_row, robot_col)]  # 从字典中删除
                        # tubes_dict[(robot_row, robot_col)] = None
                        obstacle_list.remove(tube_to_grasp)  # 从碰撞检测列表中移除
                        tube_rack_matrix = simulate_grasp(tube_rack_matrix, (robot_row, robot_col), "Robot")

                        # 打印机器人抓取后的状态矩阵
                        logging.info(f"机器人抓取位置 ({robot_row}, {robot_col}) 后的试管架状态矩阵:")
                        logging.info(str(tube_rack_matrix))
                        print(f"机器人抓取位置 ({robot_row}, {robot_col}) 后的试管架状态矩阵:")
                        print(str(tube_rack_matrix))
                        grasp_count += 1
                    else:
                        logging.warning(f"无法获取位置 ({robot_row}, {robot_col}) 的抓取路径,尝试下一个位置")
                        print(f"无法获取位置 ({robot_row}, {robot_col}) 的抓取路径,尝试下一个位置")
                        # break
                        continue
                else:
                    logging.error(
                        f"robot_grasp.grasp_tube 返回 None,无法获取位置 ({robot_row}, {robot_col}) 的抓取路径")
                    print(f"robot_grasp.grasp_tube 返回 None,无法获取位置 ({robot_row}, {robot_col}) 的抓取路径")
                    # break
                    continue
            else:
                logging.info("放弃机器人抓取.")
                print("放弃机器人抓取.")
                continue

        else:
            # 机器人无可抓取位置,需要人工干预
            logging.info("程序判断:机器人没有可以抓取的位置,需要人工干预")
            print(f"程序判断:机器人没有可以抓取的位置,需要人工干预")

            human_position = find_best_human_grasp_position(tube_rack_matrix)

            if human_position:
                human_row, human_col = human_position
                logging.info(f"需要人工抓取位置 ({human_row}, {human_col}) 的试管")
                print(f"需要人工抓取位置 ({human_row}, {human_col}) 的试管")

                action = input("是否执行人工抓取？ (y/n): ")
                if action.lower() == 'y':
                    # 保存人工抓取位置信息
                    grasp_positions.append({"row": human_row, "col": human_col, "type": "Human"})

                    # 模拟人工抓取
                    tube_rack_matrix = simulate_grasp(tube_rack_matrix, (human_row, human_col), "Human")

                    # 打印人工抓取后的状态矩阵
                    logging.info(f"人工抓取({human_row}, {human_col})后的试管架状态矩阵:")
                    logging.info(str(tube_rack_matrix))
                    print(f"人工抓取({human_row}, {human_col})后的试管架状态矩阵:")
                    print(str(tube_rack_matrix))

                    # 从环境中移除试管
                    if (human_row, human_col) in tubes_dict:
                        tube_to_remove = tubes_dict[(human_row, human_col)]
                        robot_grasp.detach_tube(tube_to_remove)
                        del tubes_dict[(human_row, human_col)]
                        # tubes_dict[(human_row, human_col)] = None
                        obstacle_list.remove(tube_to_remove)
                        grasp_count += 1
                    else:
                        logging.warning(f"位置 ({human_row}, {human_col}) 的试管已经不存在")
                        print(f"位置 ({human_row}, {human_col}) 的试管已经不存在")

                    # 寻找机器人可抓取的位置
                    robot_positions = find_robot_grabbable_positions(tube_rack_matrix)
                    if robot_positions:
                        logging.info("人工抓取后,机器人可以抓取的位置:")
                        print("人工抓取后,机器人可以抓取的位置:")
                        for pos in robot_positions:
                            logging.info(f"  - ({pos[0]}, {pos[1]})")
                            print(f"  - ({pos[0]}, {pos[1]})")
                    else:
                        logging.info("人工抓取后,机器人没有可以抓取的位置")
                        print("人工抓取后,机器人没有可以抓取的位置")

                else:
                    logging.info("放弃人工抓取.")
                    print("放弃人工抓取.")
            else:
                logging.info("没有可以抓取的试管了！")
                print("没有可以抓取的试管了！")
                break

    logging.info("所有试管都被抓取了！")
    logging.info(f"总共抓取了{grasp_count}次")
    print("所有试管都被抓取了！")
    print(f"总共抓取了{grasp_count}次")

    # 保存抓取位置信息列表到文件
    grasp_positions_filename = os.path.join(save_dir, "grasp_positions_detection.pkl")
    with open(grasp_positions_filename, 'wb') as f:
        pickle.dump(grasp_positions, f)

    # 同时保存检测信息和抓取结果的组合数据
    combined_data = {
        "original_detection_data": robot_poses_data,
        "grasp_positions": grasp_positions,
        "total_grasps": grasp_count,
        "final_matrix": tube_rack_matrix.tolist()
    }

    combined_filename = os.path.join(save_dir, "detection_and_grasp_results.json")
    with open(combined_filename, 'w') as f:
        json.dump(combined_data, f, indent=4)

    logging.info(f"成功保存抓取位置信息到: {grasp_positions_filename}")
    logging.info(f"成功保存组合结果到: {combined_filename}")
    print(f"成功保存抓取位置信息到: {grasp_positions_filename}")
    print(f"成功保存组合结果到: {combined_filename}")
    print("处理完成！")

    sys.exit()
    base.run()
