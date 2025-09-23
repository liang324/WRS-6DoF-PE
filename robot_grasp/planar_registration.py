# -*- coding:utf-8 -*-

"""
Author: Yixuan Su
Date: 2025/07/15
File: planar_registration.py
Description: 平面点云配准方案

"""
import pyransac3d as pyrsc
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def robust_plane_alignment(scene_cloud, template_cloud):
    """
    基于法向量和几何中心的高效平面对齐
    """
    # 1. 计算几何中心
    scene_center = np.asarray(scene_cloud.get_center())
    template_center = np.asarray(template_cloud.get_center())

    # 2. 计算平均法向量
    scene_normal = np.mean(np.asarray(scene_cloud.normals), axis=0)
    scene_normal /= np.linalg.norm(scene_normal)

    template_normal = np.array([0., 0., 1.])  # 假设模板法向量为Z轴

    # 3. 计算旋转矩阵(使法向量对齐)
    # 计算旋转轴和角度
    dot_product = np.dot(scene_normal, template_normal)
    if abs(dot_product - 1.0) < 1e-6:  # 法向量相同
        rotation_matrix = np.eye(3)
        print("需要反转")
    elif abs(dot_product + 1.0) < 1e-6:  # 法向量相反
        rotation_matrix = -np.eye(3)  # 180度旋转
        print("不需要反转")
    else:
        rotation_axis = np.cross(template_normal, scene_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(dot_product)
        rotation = R.from_rotvec(rotation_axis * rotation_angle)
        rotation_matrix = rotation.as_matrix()

    # 4. 平移计算(使几何中心对齐)
    translation = scene_center - np.dot(rotation_matrix, template_center)

    # 5. 构建变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation

    return transformation


def robust_plane_alignment_improved(scene_cloud, template_cloud, same_threshold_deg=5.0, opposite_threshold_deg=5.0):
    """
    基于角度阈值的平面对齐

    Args:
        scene_cloud: 场景点云
        template_cloud: 模板点云
        same_threshold_deg: 认为相同的角度阈值（度）
        opposite_threshold_deg: 认为相反的角度阈值（度）
    """
    try:
        # 1. 基础计算（同之前）
        scene_center = np.asarray(scene_cloud.get_center())
        template_center = np.asarray(template_cloud.get_center())

        scene_normal = np.mean(np.asarray(scene_cloud.normals), axis=0)
        scene_normal /= np.linalg.norm(scene_normal)

        # template_normal = np.mean(np.asarray(template_cloud.normals), axis=0)
        # template_normal /= np.linalg.norm(template_normal)
        template_normal = np.array([0., 0., 1.])

        # 2. 计算角度（更直观）
        dot_product = np.clip(np.dot(scene_normal, template_normal), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        print(f"法向量夹角: {angle_deg:.2f}度")

        # 3. 基于角度范围的判断
        if angle_deg <= same_threshold_deg:
            # 认为相同
            rotation_matrix = np.eye(3)
            print(f"法向量相同（角度{angle_deg:.2f}° ≤ {same_threshold_deg}°）")

        # elif angle_deg >= (180.0 - opposite_threshold_deg):
        #     # 认为相反
        #     rotation_matrix = -np.eye(3)
        #     print(f"法向量相反（角度{angle_deg:.2f}° ≥ {180.0 - opposite_threshold_deg}°）")

        elif angle_deg >= (180.0 - opposite_threshold_deg):
            # 认为相反 - 使用叉积计算旋转轴
            rotation_axis = np.cross(scene_normal, template_normal)
            if np.allclose(rotation_axis, 0):  # 处理两个向量平行或几乎平行的情况
                rotation_axis = np.array([1., 0., 0.])  # 选择一个任意的正交轴
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = R.from_rotvec(rotation_axis * np.pi)  # 旋转180度 (pi radians)
            rotation_matrix = rotation.as_matrix()
            print(f"法向量相反（角度{angle_deg:.2f}° ≥ {180.0 - opposite_threshold_deg}°）")

        else:
            # 需要旋转
            rotation_axis = np.cross(scene_normal, template_normal)
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(dot_product)
            rotation = R.from_rotvec(rotation_axis * rotation_angle)
            rotation_matrix = rotation.as_matrix()
            print(f"需要旋转{angle_deg:.2f}度")

        # 4. 计算平移
        rotated_template_center = np.dot(rotation_matrix, template_center)
        translation = scene_center - rotated_template_center

        # 5. 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = translation

        return transformation

    except Exception as e:
        print(f"平面对齐失败: {str(e)}")
        return np.eye(4)


def adaptive_plane_alignment(scene_cloud, template_cloud, noise_level="medium"):
    """
    自适应阈值的平面对齐

    Args:
        noise_level: "low", "medium", "high"
    """
    # 根据噪声水平设置阈值
    thresholds = {
        "low": {"same": 2.0, "opposite": 2.0},
        "medium": {"same": 5.0, "opposite": 5.0},
        "high": {"same": 10.0, "opposite": 10.0}
    }

    threshold = thresholds.get(noise_level, thresholds["medium"])

    return robust_plane_alignment_improved(
        scene_cloud, template_cloud,
        same_threshold_deg=threshold["same"],
        opposite_threshold_deg=threshold["opposite"]
    )


def icp_refinement(scene_cloud, template_cloud, initial_transformation):
    """
    ICP精配准
    """
    # 点对点ICP(粗配准)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        template_cloud, scene_cloud, 0.05, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # 点对平面ICP(精配准)
    icp_fine = o3d.pipelines.registration.registration_icp(
        template_cloud, scene_cloud, 0.05, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    return icp_fine.transformation


def visualize_registration(scene, template, transformation, title="Registration Result"):
    """
    可视化配准结果
    """
    template_temp = copy.deepcopy(template)
    scene_temp = copy.deepcopy(scene)

    # 应用变换
    template_temp.transform(transformation)

    # 为点云上色以便区分
    scene_temp.paint_uniform_color([1, 0, 0])  # 红色为场景点云
    template_temp.paint_uniform_color([0, 1, 0])  # 绿色为模板点云

    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # 可视化
    o3d.visualization.draw_geometries([scene_temp, template_temp, coord_frame], window_name=title)


def evaluate_registration(scene, template, transformation):
    """
    评估配准质量
    """
    # 计算配准误差
    template_transformed = template.transform(transformation)

    # 计算点到点的距离
    dists = scene.compute_point_cloud_distance(template_transformed)
    dists = np.asarray(dists)
    mean_error = np.mean(dists)
    max_error = np.max(dists)

    print(f"配准误差评估:")
    print(f"  平均距离误差: {mean_error:.4f} mm")
    print(f"  最大距离误差: {max_error:.4f} mm")

    # 可视化误差热力图
    error_colors = np.zeros((len(dists), 3))
    max_dist = np.max(dists) * 1.5
    error_colors[:, 0] = np.clip(dists / max_dist, 0, 1)  # 红色通道表示误差大小
    error_colors[:, 1] = np.clip(1 - dists / max_dist, 0, 1)  # 绿色通道表示匹配良好

    template_transformed.colors = o3d.utility.Vector3dVector(error_colors)

    # 可视化
    scene_temp = copy.deepcopy(scene)
    scene_temp.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色场景点云
    o3d.visualization.draw_geometries([scene_temp, template_transformed],
                                      window_name="Registration Error")


if __name__ == "__main__":
    # 加载点云数据
    scene_cloud = o3d.io.read_point_cloud(
        r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaborative_Tube_Grasping\Task3_pose_Estimation\Mech_Mind_Camera\20250611_210954\TexturedPointCloud_meters.ply")

    template_cloud = o3d.io.read_point_cloud(
        r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaborative_Tube_Grasping\Task3_pose_Estimation\AAA_template_mesh_and_pcd\centri_tube10_center\centri_tube10_center_plane.ply")

    # 获取场景中心点在相机坐标系中的位置
    scene_center = np.asarray(scene_cloud.get_center())
    print("\n场景中心点在相机坐标系中的坐标 (单位: 米):", scene_center)

    # 获取模板中心点在相机坐标系中的位置
    template_cloud_center1 = np.asarray(template_cloud.get_center())
    print("\n模板中心点在相机坐标系中的坐标 (单位: 米):", template_cloud_center1)

    # 确保点云有法向量
    if not scene_cloud.has_normals():
        scene_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))

    if not template_cloud.has_normals():
        template_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))

    # 阶段1: 平面法向量对齐
    print("=== 阶段1: 平面法向量对齐 ===")
    initial_transformation = robust_plane_alignment(scene_cloud, template_cloud)
    print("初始变换矩阵:\n", initial_transformation)
    visualize_registration(scene_cloud, template_cloud, initial_transformation, "Initial Alignment")

    # 阶段2: ICP精配准
    print("\n=== 阶段2: ICP精配准 ===")
    final_transformation = icp_refinement(scene_cloud, template_cloud, initial_transformation)
    print("精配准变换矩阵:\n", final_transformation)
    visualize_registration(scene_cloud, template_cloud, final_transformation, "Final Registration")

    # 评估配准结果
    print("\n=== 配准结果评估 ===")
    evaluate_registration(scene_cloud, template_cloud, final_transformation)

    # 获取场景中心点在相机坐标系中的位置
    scene_center = np.asarray(scene_cloud.get_center())
    print("\n场景中心点在相机坐标系中的坐标 (单位: 米):", scene_center)

    # 获取模板中心点在相机坐标系中的位置
    template_cloud_center = np.asarray(template_cloud.get_center())
    print("\n模板中心点在相机坐标系中的坐标 (单位: 米):", template_cloud_center)

    # 获取模板中心点在相机坐标系中的位置
    rack_center_in_camera_frame = (final_transformation[:3, :3] @ template_cloud_center1 +
                                   final_transformation[:3, 3])
    # 打印试管架在相机坐标系下的中心点坐标
    print("试管架在相机坐标系下的中心点坐标:", rack_center_in_camera_frame)

    # 计算 XYZ 误差
    xyz_error = scene_center - template_cloud_center
    # 设置打印格式
    np.set_printoptions(suppress=True, precision=6)
    print("\nXYZ 误差 (单位: 米):", xyz_error)
