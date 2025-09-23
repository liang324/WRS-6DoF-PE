"""
Author: Yixuan Su
Date: 2024/12/16 14:06
File: aaaaa_stl_to_pcd_uniform_sampling.py
Description:

"""
import os
import trimesh
import numpy as np
import open3d as o3d


def stl_to_point_cloud_uniform_sampling(stl_path, num_points=1000, save_path="stl_to_point_cloud_uniform_sampling.ply"):
    # 确保保存路径的目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # 加载STL模型
    mesh = trimesh.load_mesh(stl_path)

    # 从STL模型中进行均匀采样
    points = mesh.sample(num_points)

    # 转换为Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

    # 保存点云
    o3d.io.write_point_cloud(save_path, point_cloud)
    print(f"Point cloud saved to {save_path}")
    return point_cloud


# 使用方法
stl_path = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\refrigerator_open.STL"
save_path = r"D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_AG145\objects\refrigerator_open.ply"
point_cloud = stl_to_point_cloud_uniform_sampling(stl_path, 500000, save_path)
