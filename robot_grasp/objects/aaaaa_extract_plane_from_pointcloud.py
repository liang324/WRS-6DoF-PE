#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :icp
@File :aaaaa_extract_plane_from_pointcloud.py
@IDE :PyCharm
@Author  ：Yixuan Su
@Date :2025/6/10 10:40
Description:

'''

import os
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import copy


class PointCloudSelector:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)
        self.selected_points = []
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window(window_name="选择三个点定义平面")

    def select_points(self):
        """用户交互式选择点"""
        print("请按住shift+左键选择三个点定义平面,按Q结束选择")
        self.vis.add_geometry(self.pcd)
        self.vis.run()  # 阻塞直到窗口关闭
        self.vis.destroy_window()
        picked_points = self.vis.get_picked_points()

        if len(picked_points) < 3:
            print("错误:至少需要选择三个点定义平面")
            return None

        return [self.points[i] for i in picked_points][:3]  # 只取前三个点


def calculate_plane_from_points(points):
    """根据三个点计算平面方程"""
    A = np.array(points[0])
    B = np.array(points[1])
    C = np.array(points[2])

    # 计算平面法向量
    normal = np.cross(B - A, C - A)
    normal = normal / np.linalg.norm(normal)  # 归一化

    # 计算平面常数项
    d = -np.dot(normal, A)
    return np.append(normal, d)  # [a, b, c, d]


def extract_plane(pcd_path, save_dir_plane, save_dir_remaining, threshold=0.002, max_iterations=10000,
                  interactive=True):
    """
    从点云中提取平面并保存
    :param pcd_path: 点云路径
    :param save_dir_plane: 平面点云保存路径
    :param save_dir_remaining: 剩余点云保存路径
    :param threshold: 点到平面的最大距离阈值
    :param max_iterations: RANSAC最大迭代次数
    :param interactive: 是否启用交互式选点
    """
    # 1. 创建保存目录
    os.makedirs(save_dir_plane, exist_ok=True)
    os.makedirs(save_dir_remaining, exist_ok=True)

    # 2. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 3. 交互式平面检测
    if interactive:
        print("===== 交互式平面分割模式 =====")
        selector = PointCloudSelector(pcd)
        selected_points = selector.select_points()

        if not selected_points or len(selected_points) < 3:
            print("平面定义失败,使用默认RANSAC方法")
            interactive = False
        else:
            # 计算平面方程
            plane_model = calculate_plane_from_points(selected_points)

            # 可视化用户选择的点
            selected_pcd = o3d.geometry.PointCloud()
            selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
            selected_pcd.paint_uniform_color([1, 0, 0])  # 红色显示选中的点

            # 计算所有点到平面的距离
            distances = np.abs(np.dot(points, plane_model[:3]) + plane_model[3])
            inliers = np.where(distances < threshold)[0]

    # 4. 非交互模式使用RANSAC
    if not interactive:
        print("===== 使用RANSAC平面分割 =====")
        plane = pyrsc.Plane()
        plane_model, inliers = plane.fit(points, thresh=threshold, maxIteration=max_iterations)
        print(f"检测到平面方程: {plane_model}")

    # 5. 分割点云
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # 6. 保存点云
    filename = os.path.basename(pcd_path).split(".")[0]
    plane_filename = os.path.join(save_dir_plane, f"{filename}_plane.ply")
    remaining_filename = os.path.join(save_dir_remaining, f"{filename}_remaining.ply")

    o3d.io.write_point_cloud(plane_filename, inlier_cloud)
    o3d.io.write_point_cloud(remaining_filename, outlier_cloud)

    print(f"平面点云保存至: {plane_filename}")
    print(f"剩余点云保存至: {remaining_filename}")

    # 7. 可视化结果
    inlier_cloud.paint_uniform_color([1, 0, 0])  # 平面点云设为红色
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 剩余点云设为灰色

    if interactive:
        # 显示用户选择的点
        o3d.visualization.draw_geometries([selected_pcd, inlier_cloud], window_name="交互式平面分割结果")
        o3d.visualization.draw_geometries([selected_pcd, outlier_cloud], window_name="交互式平面分割结果")
        o3d.visualization.draw_geometries([selected_pcd, inlier_cloud, outlier_cloud], window_name="交互式平面分割结果")
    else:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="RANSAC平面分割结果")


if __name__ == '__main__':
    # 设置参数
    pcd_path = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\refrigerator_open.ply"
    save_dir_plane = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects"
    save_dir_remaining = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects"

    # 提取平面（启用交互模式）
    extract_plane(pcd_path, save_dir_plane, save_dir_remaining, interactive=True)
