#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：WRS-6DoF-PE
@File    ：local_detection_utils.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""
import pyransac3d as pyrsc
import argparse
from collections import namedtuple
import pyrealsense2 as rs
from robot_grasp import YOLOv9_Detect_API
from robot_grasp.simulation_utils import sam_segment, save_object_mask_on_black_background, \
    extract_point_cloud, extract_tube_point_clouds, SimplifiedTubePoseEstimator, \
    select_largest_object, icp_plane_estimate_pose, extract_plane
from robot_grasp.azure_kinect_recorder import RecorderWithCallback
import os.path
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import robot_con.gofa_con.gofa_con as gofa_con
import drivers.devices.dh.maingripper as dh_r
import copy
import json
import sys
from robot_grasp.YOLOv9_Detect_API import DetectAPI
from camera_utils import capture_point_cloud
import shutil
import os
import cv2
import numpy as np
import open3d as o3d
import os.path
import modeling.collision_model as cm
import visualization.panda.world as wd
import robot_sim.robots.gofa5.gofa5_Ag145 as gf5


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


def transform_camera_to_robot(camera_xyz, T):
    """相机坐标 -> 机器人坐标"""
    p_c = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0])
    p_r = T @ p_c
    return np.round(p_r[:3], 3)  # 取前三个分量 & 保留3位小数


def print_detected_objects(results):
    print("\n=== 检测到的物体列表 ===")
    for idx, obj in enumerate(results):
        pos = np.round(obj["robot_xyz_m"], 3)
        print(f"{idx}: {obj['class_name']:<10} 位置: {pos}")


class RobotController:
    def __init__(self):
        self.base, self.rbt_s, self.rbt_r, self.grip_r, self.rrtc_s, self.ppp_s = self.init_robot_system()

    def init_robot_system(self, open_width=0.023):
        base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
        rbt_s = gf5.GOFA5()
        rbt_r = gofa_con.GoFaArmController()
        grip_r = dh_r.MainGripper('com3', speed=100)
        rrtc_s_ = rrtc.RRTConnect(rbt_s)
        ppp_s_ = ppp.PickPlacePlanner(rbt_s)
        rbt_s.hnd.jaw_to(open_width / 2)
        grip_r.jaw_to(open_width / 2)

        return base, rbt_s, rbt_r, grip_r, rrtc_s_, ppp_s_

    def go_init(self):
        """回到机械臂初始位置,成功 True / 失败 False"""
        try:
            init_jnts_values = np.array([0, 0, 0, 0, 0, 0])
            current_jnts = self.rbt_r.get_jnt_values()
            print("[go_init] 当前关节位置:", current_jnts)

            init_path = self.rrtc_s.plan(
                component_name="arm",
                start_conf=current_jnts,
                goal_conf=init_jnts_values,
                ext_dist=0.01,
                max_time=300
            )

            if init_path is None:
                print("[go_init] 初始化路径规划失败")
                return False

            self.rbt_r.move_jntspace_path(init_path)
            print("[go_init] 初始化位置完成.")
            return True

        except Exception as e:
            print(f"[go_init] 执行失败: {e}")
            return False

    def go_place(self):
        """放置操作,成功 True / 失败 False"""
        try:
            init_jnts_values = np.array([1.38055544, 0.00628319, 0., -0., -0., -0.])
            current_jnts = self.rbt_r.get_jnt_values()
            print("[go_place] 当前关节位置:", current_jnts)

            place_path = self.rrtc_s.plan(
                component_name="arm",
                start_conf=current_jnts,
                goal_conf=init_jnts_values,
                ext_dist=0.01,
                max_time=300
            )

            if place_path is None:
                print("[go_place] 放置路径规划失败")
                return False

            self.rbt_r.move_jntspace_path(place_path)
            print("[go_place] 放置位置完成.")
            return True

        except Exception as e:
            print(f"[go_place] 执行失败: {e}")
            return False

    def move_to_object(self, target_obj, move_real=False, obstacles=None):
        """执行移动到目标物体上方,成功返回 True,失败返回 False"""

        try:
            target_pos = np.array(target_obj["robot_xyz_m"])
            print(f"[INFO] 移动到: {target_obj['class_name']} 位置: {target_pos}")

            # # 上方偏移位置
            # approach_height = 0.1
            # approach_pos = target_pos + np.array([-0.1, 0, approach_height])

            # ==== 高度映射表 ====
            approach_height_map = {
                "ore": 0.25,  # refrigerator_open
                "cre": 0.25,  # refrigerator_yuan
                "oce": 0.15,  # centrifuge_machine
                "cce": 0.15,  # centrifuge_machine
                "locker": 0.30,  # locker
                "rack510": 0.20,  # rack510
                "rack34": 0.05,  # rack34
                "unknown": 0.1  # 默认
            }

            # 获取偏移高度
            approach_height = approach_height_map.get(target_obj['class_name'], approach_height_map["unknown"])

            # 上方偏移位置
            approach_pos = target_pos + np.array([-0.1, 0, approach_height])

            rot_app = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])

            # 直接构建 TCP 位姿矩阵
            T_tcp_in_robot = np.eye(4)
            T_tcp_in_robot[:3, :3] = rot_app
            T_tcp_in_robot[:3, 3] = approach_pos

            # # IK 求解
            # app_jnt_values = self.rbt_s.ik(tgt_pos=approach_pos, tgt_rotmat=rot_app)
            # if app_jnt_values is None:
            #     print("[WARN] IK 求解失败")
            #     return False, None

            # ==== IK 求解 + 重试 ====
            max_ik_attempts = 5
            app_jnt_values = None
            for retry in range(max_ik_attempts):
                app_jnt_values = self.rbt_s.ik(tgt_pos=approach_pos, tgt_rotmat=rot_app)
                if app_jnt_values is not None:
                    print(f"[GOOD] IK 求解成功 (尝试 {retry + 1}/{max_ik_attempts})")
                    break
                else:
                    print(f"[WARN] IK 求解失败 (尝试 {retry + 1}/{max_ik_attempts})，调整高度重试...")
                    approach_pos[2] += 0.02  # 增加 2cm 高度再试
                    T_tcp_in_robot[:3, 3] = approach_pos

            if app_jnt_values is None:
                print("[ERROR] 所有 IK 尝试均失败！")
                return False, None

            # 路径规划
            current_jnts = self.rbt_r.get_jnt_values()

            # 重试次数
            max_attempts = 100
            path_to_target = None

            for attempt in range(1, max_attempts + 1):
                print(f"[INFO] 第 {attempt} 次路径规划尝试...")
                path_to_target = self.rrtc_s.plan(component_name="arm",
                                                  start_conf=current_jnts,
                                                  goal_conf=app_jnt_values,
                                                  ext_dist=0.01,
                                                  max_time=300,
                                                  obstacle_list=obstacles
                                                  )
                if path_to_target is not None:
                    print("[GOOD] 路径规划成功")
                    break
                else:
                    print(f"[WARN] 第 {attempt} 次路径规划失败")

            if path_to_target is None:
                print("[ERROR] 路径规划100次均失败！")
                return False, None

            # 执行
            if move_real:
                self.rbt_r.move_jntspace_path(path_to_target)
            else:
                # 在仿真中显示轨迹
                for i, pose in enumerate(path_to_target):
                    if i % 15 == 0:
                        self.rbt_s.fk(component_name="arm", jnt_values=pose)
                        self.rbt_s.gen_meshmodel().attach_to(self.base)
            return True, T_tcp_in_robot
        except Exception as e:
            print(f"[ERROR] 移动失败: {e}")
            return False, None

    @staticmethod
    def get_object_configs(base_dir=None):
        """返回固定的模型路径和颜色映射表"""
        if base_dir is None:
            base_dir = r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"

        object_model_map = {
            "ore": os.path.join(base_dir, "refrigerator_open.STL"),
            "cre": os.path.join(base_dir, "refrigerator.STL"),
            "oce": os.path.join(base_dir, "centrifuge_machine.STL"),
            "cce": os.path.join(base_dir, "centrifuge_machine.STL"),
            "locker": os.path.join(base_dir, "locker.STL"),
            "rack510": os.path.join(base_dir, "rack510.STL"),
            "rack34": os.path.join(base_dir, "rack34.STL")
        }

        object_color_map = {
            "ore": [1, 0, 0, 1],  # 红色
            "cre": [1, 0, 0, 1],  # 红色
            "oce": [0.5, 0.5, 0.5, 1],  # 灰色
            "cce": [0.5, 0.5, 0.5, 1],  # 灰色
            "locker": [1, 0.5, 0, 1],  # 橙色
            "rack510": [1, 1, 1, 1],  # 白色
            "rack34": [0.5, 0.8, 1, 1],  # 淡蓝色
        }

        return object_model_map, object_color_map

    def build_obstacles(self, results, object_model_map, object_color_map):
        """根据检测结果构建障碍物"""
        obstacles = []
        rotmat = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        for obj in results:
            cname = obj["class_name"]
            if cname in object_model_map and os.path.exists(object_model_map[cname]):
                model_path = object_model_map[cname]
                cm_obj = cm.CollisionModel(model_path)
                cm_obj.set_pos(obj["robot_xyz_m"])
                cm_obj.set_rotmat(rotmat)
                cm_obj.set_rgba(object_color_map.get(cname, [1, 1, 1, 0.8]))
                cm_obj.attach_to(self.base)
                obstacles.append(cm_obj)

        self.obstacles = obstacles
        return obstacles

    def auto_move_all_objects(self, results, move_real_flag, obstacles):
        """自动依次移动到每个检测物体上方"""
        # 获取当前.py文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 数据集根目录
        base_folder = os.path.join(current_dir, "Realsense_datasets")

        clear_folder(base_folder)
        os.makedirs(base_folder, exist_ok=True)

        for idx, target_obj in enumerate(results):
            print(f"\n[INFO] 开始移动到第 {idx + 1} 个物体: {target_obj['class_name']}")

            while True:
                move_ok, T_tcp_in_robot = self.move_to_object(target_obj, move_real=move_real_flag, obstacles=obstacles)

                if move_ok:
                    # 移动成功后执行拍摄或位姿估计
                    print("[INFO] 开始拍摄和位姿估计...")
                    # === 为当前类创建保存路径 ===
                    obj_class = target_obj['class_name']
                    obj_dirs, object_root = prepare_object_save_dirs(base_folder, obj_class)

                    if obj_class == "ore":
                        while True:
                            print("[INFO] ===  开始执行 二次检测阶段 拍摄任务 ===")
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])

                            print("[INFO] === 开始执行 Refrigerator open 位姿重定位任务 ===")
                            open_detector = RefrigeratorDetector(mode="open", weights_path="best.pt")
                            Refrigerator_result = open_detector.process_refrigerator_from_image(
                                color_img=color_image,
                                depth_img=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                mask_dir=obj_dirs["masks"],
                                pointcloud_dir=obj_dirs["point_clouds"],
                                plane_dir=obj_dirs["planes"],
                                remaining_dir=obj_dirs["remaining"]
                            )
                            print("[INFO] Step 1: Refrigerator Open 检测完成")

                            if Refrigerator_result is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                # 保存结果
                                open_detector.save_detection_result(Refrigerator_result, save_folder=object_root)
                                print("T_Open_refrigerator_in_robot:\n", Refrigerator_result.T_refrigerator_in_robot)
                                break

                    elif obj_class == "cre":
                        while True:
                            print("[INFO] ===  开始执行 二次检测阶段 拍摄任务 ===")
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])

                            print("[INFO] === 开始执行 Refrigerator close 位姿重定位任务 ===")
                            close_detector = RefrigeratorDetector(mode="close", weights_path="best.pt")
                            Refrigerator_result = close_detector.process_refrigerator_from_image(
                                color_img=color_image,
                                depth_img=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                mask_dir=obj_dirs["masks"],
                                pointcloud_dir=obj_dirs["point_clouds"],
                                plane_dir=obj_dirs["planes"],
                                remaining_dir=obj_dirs["remaining"]
                            )
                            print("[INFO] Step 1: Refrigerator close 检测完成")

                            if Refrigerator_result is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                # 保存结果
                                close_detector.save_detection_result(Refrigerator_result, save_folder=object_root)
                                print("T_CLose_refrigerator_in_robot:\n", Refrigerator_result.T_refrigerator_in_robot)
                                break

                    elif obj_class == "oce":
                        while True:
                            print("[INFO] ===  开始执行 二次检测阶段 拍摄任务 ===")
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])
                            print("[INFO] === 开始执行 Centrifuge open 位姿重定位任务 ===")
                            open_detector = CentrifugeDetector(mode="open", weights_path="best.pt")
                            Centrifuge_result = open_detector.process_centrifuge_from_image(
                                color_img=color_image,
                                depth_img=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                mask_dir=obj_dirs["masks"],
                                pointcloud_dir=obj_dirs["point_clouds"],
                                plane_dir=obj_dirs["planes"],
                                remaining_dir=obj_dirs["remaining"]
                            )
                            print("[INFO] Step 1: Centrifuge open 检测完成")

                            if Centrifuge_result is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                # 保存结果
                                open_detector.save_detection_result(Centrifuge_result, save_folder=object_root)
                                print("T_Open_centrifuge_in_robot:\n", Centrifuge_result.T_centrifuge_in_robot)
                                break

                    elif obj_class == "cce":
                        while True:
                            print("[INFO] ===  开始执行 二次检测阶段 拍摄任务 ===")
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])
                            print("[INFO] === 开始执行 Centrifuge close 位姿重定位任务 ===")
                            close_detector = CentrifugeDetector(mode="close", weights_path="best.pt")
                            Centrifuge_result = close_detector.process_centrifuge_from_image(
                                color_img=color_image,
                                depth_img=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                mask_dir=obj_dirs["masks"],
                                pointcloud_dir=obj_dirs["point_clouds"],
                                plane_dir=obj_dirs["planes"],
                                remaining_dir=obj_dirs["remaining"]
                            )
                            print("[INFO] Step 1: CentrifugeDetector close 检测完成")

                            if Centrifuge_result is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                # 保存结果
                                close_detector.save_detection_result(Centrifuge_result, save_folder=object_root)
                                print("T_Close_centrifuge_in_robot:\n", Centrifuge_result.T_centrifuge_in_robot)
                                break

                    elif obj_class == "locker":
                        while True:
                            print("[INFO] ===  开始执行 二次检测阶段 拍摄任务 ===")
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])
                            print("[INFO] === 开始执行 Locker 位姿重定位任务 ===")
                            Locker_detector = LockerDetector(weights_path="best.pt")
                            Locker_result = Locker_detector.process_locker_from_image(
                                color_img=color_image,
                                depth_img=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                mask_dir=obj_dirs["masks"],
                                pointcloud_dir=obj_dirs["point_clouds"],
                                plane_dir=obj_dirs["planes"],
                                remaining_dir=obj_dirs["remaining"]
                            )
                            print("[INFO] Step 1: Locker 检测完成")

                            if Locker_result is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                # 保存结果
                                Locker_detector.save_detection_result(Locker_result, save_folder=object_root)
                                print("T_locker_in_robot:\n", Locker_result.T_locker_in_robot)
                                break

                    elif obj_class == "rack510":
                        while True:
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])

                            # rack510 特定处理（含试管架 + 试管处理）
                            pipeline = Rack510Pipeline(weights_path="best_rack510.pt",
                                                       rack_shape=(5, 10),
                                                       center_holes=[(2, 4), (2, 5)],
                                                       tube_spacing=0.025,
                                                       tube_diameter=0.021
                                                       )
                            rack_and_tubes_data = pipeline.run(
                                color_image=color_image,
                                depth_image=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                save_dirs={"masks": obj_dirs["masks"],
                                           "point_clouds": obj_dirs["point_clouds"],
                                           "planes": obj_dirs["planes"],
                                           "remaining": obj_dirs["remaining"],
                                           "pcd": obj_dirs["pcd"]}
                            )

                            if rack_and_tubes_data is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                rack_matrix = rack_and_tubes_data["rack_info"]["rack_matrix"]
                                print("Rack matrix:", rack_matrix)
                                for tube in rack_and_tubes_data["tubes_info"]:
                                    print(f"Tube {tube['label']} pose in robot:\n", tube["Tube_center_pose_in_robot"])
                                break

                    elif obj_class == "rack34":
                        while True:
                            color_image, save_colorpath, depth_image, save_depthpath, original_point_cloud, save_pcdpath = capture_point_cloud(
                                obj_dirs["pcd"])

                            # rack510 特定处理（含试管架 + 试管处理）
                            pipeline = Rack34Pipeline(weights_path="best_rack34.pt",
                                                      rack_shape=(3, 4),
                                                      center_holes=[(1, 1), (1, 2)],
                                                      tube_spacing=0.03,
                                                      tube_diameter=0.021)

                            rack_and_tubes_data = pipeline.run(
                                color_image=color_image,
                                depth_image=depth_image,
                                T_tcp_in_robot=T_tcp_in_robot,
                                save_dirs={"masks": obj_dirs["masks"],
                                           "point_clouds": obj_dirs["point_clouds"],
                                           "planes": obj_dirs["planes"],
                                           "remaining": obj_dirs["remaining"],
                                           "pcd": obj_dirs["pcd"]}
                            )

                            if rack_and_tubes_data is None:
                                print(f"[WARN] {obj_class} 检测失败！")
                                retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                                if retry == "y":
                                    print("[INFO] 重新拍摄并检测中...")
                                    continue  # 重新回到 while True 顶部,重新执行拍摄与检测
                                else:
                                    print(f"[INFO] 用户选择跳过 {obj_class}")
                                    break  # 跳出 while True,去处理下一个物体
                            else:
                                rack_matrix = rack_and_tubes_data["rack_info"]["rack_matrix"]
                                print("Rack matrix:", rack_matrix)
                                for tube in rack_and_tubes_data["tubes_info"]:
                                    print(f"Tube {tube['label']} pose in robot:\n", tube["Tube_center_pose_in_robot"])
                                break
                    else:
                        print(f"[WARN] 未定义 {obj_class} 的特定处理逻辑,跳过.")
                        break

                    print(f"[INFO] {target_obj['class_name']} 数据保存完成至 {obj_dirs}")
                    break
                else:
                    print(f"[WARN] 移动到 {target_obj['class_name']} 失败,不能进行二次相机的检测")
                    retry = input(f"是否重新规划并移动到 {target_obj['class_name']}？(y/n): ").strip().lower()
                    if retry == "y":
                        print(f"[INFO] 正在重新规划路径到 {target_obj['class_name']} ...")
                        continue
                    else:
                        print(f"[INFO] 用户选择跳过{target_obj['class_name']}")
                        break

    def shutdown(self):
        """关闭系统"""
        self.base.run()
        print("[RobotController] 已关闭.")


def prepare_object_save_dirs(base_folder, class_name):
    """
    创建按物体类别划分的保存目录,并返回各个子目录路径
    :param class_name: 物体类别名,例如 'rack510'
    :return: dict 保存所有路径
    """
    # 当前物体的主文件夹
    object_root = os.path.join(base_folder, class_name)

    # 定义所需的子文件夹
    subfolders = {
        "pcd": os.path.join(object_root, "pcd"),
        "masks": os.path.join(object_root, "masks"),
        "yolo_detection_image": os.path.join(object_root, "yolo_detections.jpg"),
        "point_clouds": os.path.join(object_root, "point_clouds"),
        "planes": os.path.join(object_root, "planes"),
        "remaining": os.path.join(object_root, "remaining"),
        "registration": os.path.join(object_root, "registration")
    }

    # 创建目录
    for key, folder in subfolders.items():
        if key != "yolo_detection_image":
            os.makedirs(folder, exist_ok=True)

    return subfolders, object_root


def capture_kinect_data(save_folder, config_file='default_config.json'):
    """调用 Azure Kinect 采集彩色、深度、点云数据（人工触发）,返回是否成功+文件路径"""

    clear_folder(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description='Azure Kinect 数据采集')
    parser.add_argument('--config', default=config_file, type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--align_depth_to_color', default=True)
    parser.add_argument('--output', default='kinect_data_global', type=str)
    parser.add_argument('--save_ply', default=True)
    args = parser.parse_args()

    # 读取 Kinect 配置
    config = o3d.io.read_azure_kinect_sensor_config(args.config)
    current_dir_path = os.path.abspath(os.path.dirname(__file__))

    # 创建录制器
    r = RecorderWithCallback(config, args.device, current_dir_path, args.align_depth_to_color, args)

    # 获取模式
    mode = input("输入录制模式, manual 或 auto: ")
    record_fps = float(input("输入录制帧率: ")) if mode == "auto" else -1
    print("record_fps", record_fps)

    color_path, depth_path, ply_path = r.run(mode, record_fps, args.output)
    r.recorder.close_record()

    # ====== 成功判断 ======
    # 检查文件是否存在且不为空
    if (color_path and os.path.exists(color_path) and os.path.getsize(color_path) > 0 and
            depth_path and os.path.exists(depth_path) and os.path.getsize(depth_path) > 0):
        print("[INFO] 拍摄成功")
        return True, color_path, depth_path

    print("[ERROR] 拍摄失败,未生成有效文件")
    return False, None, None


def setup_kinectv3_camera_params():
    """返回相机内参、外参矩阵以及深度比例"""
    Kinect_data_savefolder = r'D:\AI\WRS-6DoF-PE\robot_grasp\kinect_data_global'

    fx = 1957.6904296875
    fy = 1957.29443359375
    ppx = 2042.3841552734375
    ppy = 1558.51318359375
    depth_scale = 0.001

    depth_intrin = rs.intrinsics()
    depth_intrin.width = 4096
    depth_intrin.height = 3072
    depth_intrin.ppx = ppx
    depth_intrin.ppy = ppy
    depth_intrin.fx = fx
    depth_intrin.fy = fy
    depth_intrin.model = rs.distortion.none
    depth_intrin.coeffs = [0, 0, 0, 0, 0]

    T_camera_to_robot = np.array([
        [-0.017450, -0.999249, -0.034590, 0.550000],
        [-0.999695, 0.016833, 0.018048, -0.020000],
        [-0.017452, 0.034894, -0.999239, 1.470000],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])

    return depth_intrin, T_camera_to_robot, depth_scale, Kinect_data_savefolder


def detect_and_localize(color_path, depth_path, depth_intrin, T_camera_to_robot, depth_scale=0.001):
    """运行YOLO检测并转换为机器人坐标系,并根据固定高度表修正Z坐标"""
    img_color = cv2.imread(color_path)
    img_depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 深度转换为米
    img_depth = img_depth_raw * depth_scale if img_depth_raw.dtype == np.uint16 else img_depth_raw.astype(np.float32)

    # 裁剪区域（只检测部分区域）
    crop_x1, crop_y1, crop_x2, crop_y2 = 925, 710, 3040, 2400
    img_color_crop = img_color[crop_y1:crop_y2, crop_x1:crop_x2]

    # 加载 YOLO 模型
    model = YOLOv9_Detect_API.DetectAPI(weights='abb_best.pt')
    model.opt.conf_thres = 0.5
    _, pred, _ = model.detect([img_color_crop])

    # 固定高度表（单位：米）
    object_distance = {
        "ore": 0.35 - 0.015,  # 0.335
        "cre": 0.35 - 0.015,  # 0.335
        "oce": 0.21 - 0.015,  # 0.195
        "cce": 0.21 - 0.015,  # 0.195
        "locker": 0.31 - 0.015,  # 0.300
        "rack510": 0.1 - 0.015,  # 0.085
        "rack34": 0.12 - 0.015,  # 0.105
    }

    results = []
    if pred is not None and len(pred):
        for det in pred:
            for *xyxy, conf, cls in det:
                # 裁剪坐标 → 原图坐标
                x_min = int(xyxy[0] + crop_x1)
                y_min = int(xyxy[1] + crop_y1)
                x_max = int(xyxy[2] + crop_x1)
                y_max = int(xyxy[3] + crop_y1)

                # 中心像素
                ux, uy = (x_min + x_max) // 2, (y_min + y_max) // 2

                # 获取深度
                dis = float(img_depth[uy, ux])

                # 相机坐标
                camera_xyz = np.array(
                    rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                )
                camera_xyz = np.round(camera_xyz, 3)

                # 转换到机器人坐标系
                robot_xyz = transform_camera_to_robot(camera_xyz, T_camera_to_robot)

                # 类别名
                class_name = model.names[int(cls)]

                # 初始化姿态矩阵
                robot_pose = np.eye(4)
                robot_pose[:3, 3] = robot_xyz

                # 如果有固定高度,则修正Z值
                if class_name in object_distance:
                    robot_xyz[2] = object_distance[class_name]
                    robot_pose[2, 3] = object_distance[class_name]

                # 存储结果
                results.append({
                    "class_name": class_name,
                    "confidence": round(float(conf), 3),
                    "pixel_center": [ux, uy],
                    "camera_xyz_m": camera_xyz.tolist(),
                    "robot_xyz_m": robot_xyz.tolist(),
                    "robot_pose_m": robot_pose.tolist()
                })

                # 绘制视觉效果
                cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img_color, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(img_color, (ux, uy), 5, (255, 255, 255), -1)
                coord_label = f"C:[{camera_xyz[0]} {camera_xyz[1]} {camera_xyz[2]}] R:[{robot_xyz[0]} {robot_xyz[1]} {robot_xyz[2]}]"
                cv2.putText(img_color, coord_label, (ux + 10, uy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return results, img_color


def save_detection_results(results, img_color, save_folder):
    save_img_path = os.path.join(save_folder, "detection_with_robot_coords.png")
    save_json_path = os.path.join(save_folder, "detection_results.json")
    cv2.imwrite(save_img_path, img_color)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[保存] 检测图像: {save_img_path}")
    print(f"[保存] 检测JSON: {save_json_path}")


# def execute_robot_with_command(robot_ctrl):
#     """
#     机械臂任务执行器（前端指令版）
#     严格顺序：必须初始化成功 -> 才能放置
#     指令说明：
#         00 -> 初始化
#         11 -> 初始化（重试）
#         01 -> 放置
#         12 -> 放置（重试）
#         q  -> 退出
#     返回:
#         init_done (bool): 初始化是否完成
#         place_done (bool): 放置是否完成
#     """
#     init_done = False
#     place_done = False
#
#     while True:
#         cmd = input("请输入指令 (00=初始化, 11=放置, 01,12=初始化退出): ").strip()
#
#         # 退出任务
#         if cmd in ["01", "12"]:
#             print("[INFO] 用户终止任务")
#             break
#
#         # 初始化
#         if cmd in ["00"]:
#             print("[INFO] 执行初始化动作...")
#             if robot_ctrl.go_init():
#                 init_done = True
#                 print("[INFO] 初始化成功")
#
#             else:
#                 retry = input("初始化失败,是否重试？(y/n): ").strip().lower()
#                 if retry != "y":
#                     print("[WARN] 用户放弃初始化")
#                     init_done = False
#
#         # 放置
#         elif cmd in ["11"]:
#             if not init_done:
#                 print("[ERROR] 必须先完成初始化才能执行放置")
#                 continue
#             print("[INFO] 执行放置动作...")
#             if robot_ctrl.go_place():
#                 place_done = True
#                 print("[INFO] 放置成功")
#                 return init_done, place_done
#             else:
#                 retry = input("放置失败,是否重试？(y/n): ").strip().lower()
#                 if retry != "y":
#                     print("[WARN] 用户放弃放置")
#                     place_done = False
#         else:
#             print("[WARN] 无效指令,请重新输入 (00/11=初始化, 01/12=放置, q=退出)")
#
#     return init_done, place_done


def execute_robot_with_command(robot_ctrl, cmd, init_done=False):
    """
    机械臂任务执行器（前端指令版）
    严格顺序：必须初始化成功 -> 才能放置
    指令说明：
        00 -> 初始化
        11 -> 放置
        q  -> 退出
    参数:
        robot_ctrl : RobotController 实例
        cmd (str)  : 前端传来的指令
        init_done (bool) : 记录是否已初始化（默认 False）
    返回:
        init_done (bool): 初始化是否完成
        place_done (bool): 放置是否完成
        message (str): 执行过程说明
    """
    place_done = False
    message = ""

    # 退出任务
    if cmd == "q":
        message = "[INFO] 用户终止任务"
        return init_done, place_done, message

    # 初始化
    if cmd == "00":
        message = "[INFO] 执行初始化动作..."
        if robot_ctrl.go_init():
            init_done = True
            message += " 初始化成功"
        else:
            init_done = False
            message += " 初始化失败"

    # 放置
    elif cmd == "11":
        if not init_done:
            message = "[ERROR] 必须先完成初始化才能执行放置"
            return init_done, place_done, message
        message = "[INFO] 执行放置动作..."
        if robot_ctrl.go_place():
            place_done = True
            message += " 放置成功"
        else:
            place_done = False
            message += " 放置失败"

    else:
        message = "[WARN] 无效指令，必须是 00 / 11 / q"

    return init_done, place_done, message


def kinectv3_capture_and_detect(save_folder, depth_intrin, T_camera_to_robot, depth_scale):
    """
    拍摄 + 目标检测循环流程
    :return: results, img_color
    """
    while True:  # 总循环: 拍摄 + 检测
        # ===== 拍摄 =====
        while True:
            success, color_path, depth_path = capture_kinect_data(save_folder)
            if success:
                print("[INFO] 拍摄成功")
                break
            else:
                retry = input("拍摄失败,是否重新拍摄？(y/n): ").strip().lower()
                if retry == "y":
                    print("[INFO] 重新拍摄...")
                    continue
                else:
                    print("[WARN] 用户终止拍摄,退出任务")
                    sys.exit(1)

        # ===== 目标检测 =====
        if color_path and depth_path:
            print("[INFO] 开始目标检测...")
            results, img_color = detect_and_localize(
                color_path,
                depth_path,
                depth_intrin,
                T_camera_to_robot,
                depth_scale
            )

            if results and len(results) > 0:
                print(f"[INFO] 目标检测完成,共检测到 {len(results)} 个目标")
                save_detection_results(results, img_color, save_folder)
                return results, img_color  # 成功 → 返回结果
            else:
                print("[WARN] 未检测到目标")
                retry_detect = input("是否重新拍摄并检测？(y/n): ").strip().lower()
                if retry_detect == "y":
                    print("[INFO] 重新开始拍摄+检测流程...")
                    continue  # 回到总循环
                else:
                    print("[ERROR] 用户终止任务")
                    sys.exit(1)
        else:
            print("[ERROR] 没有有效拍摄数据,无法进行目标检测")
            sys.exit(1)


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
    DEFAULT_WEIGHTS_PATH = "best_rack510.pt"

    def __init__(self,
                 weights_path=None,
                 sam_checkpoint="D:\AI\WRS-6DoF-PE\weights\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 use_new_intrinsics=False,
                 template_dir=None):
        """初始化固定资源"""
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.intrinsics = CAMERA_INTRINSICS_NEW if use_new_intrinsics else CAMERA_INTRINSICS
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH

        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"
        self.template_cloud_rack510 = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack510_plane.ply"))
        self.template_cloud_rack510_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack510.ply"))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_T_rack510_in_robot(self,
                                   T_tcp_in_robot,
                                   T_cam_in_flange=None,
                                   T_tcp_offset=None,
                                   T_rack_in_cam=None):

        """
        计算试管架在机器人坐标系下的位姿

        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_rack_in_cam: (4x4) 试管架510 在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) T_rack510_in_robot
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

        # 最终 T_rack510_in_robot
        T_rack510_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_rack_in_cam
        print("[DEBUG] T_rack510_in_robot:\n", T_rack510_in_robot)

        return T_rack510_in_robot, T_flange_in_robot, T_cam_in_flange

    def process_rack510_from_image(self,
                                   color_img, depth_img,
                                   T_tcp_in_robot,
                                   mask_dir, pointcloud_dir,
                                   plane_dir, remaining_dir,
                                   visualize=True):

        """
        从 RGB + 深度图像中检测并配准 rack510,返回 T_rack510_in_cam

        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred_results: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: T_rack510_in_cam (4x4 numpy array) 或 None
        """

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(plane_dir, exist_ok=True)
        os.makedirs(remaining_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # ---- 1. 找 rack510 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results, self.detect_api,
                                                                                         "rack510",
                                                                                         check_rotate=True)
        if largest_rack is None:
            print("[ERROR] 未检测到 rack510")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(
            point_cloud_path,
            plane_dir, remaining_dir,
            0.002, 20000
        )

        # ---- 5. ICP 位姿估计 ----
        T_rack510_in_cam = icp_plane_estimate_pose(plane_cloud, self.template_cloud_rack510, do_icp_refine=True)
        print("试管架在相机坐标系下的位姿：\n", T_rack510_in_cam)
        if T_rack510_in_cam is None:
            print("[ERROR] ICP 位姿估计失败")
            return None

            # ---- 6. 可视化（可选） ----
        if visualize:
            template_cloud_rack510_full = copy.deepcopy(self.template_cloud_rack510_full)
            aligned_scene_pcd = template_cloud_rack510_full.transform(T_rack510_in_cam)

            # 给颜色区分
            point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
            aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

            o3d.visualization.draw_geometries(
                [aligned_scene_pcd, point_cloud],
                window_name="After ICP Registration",
                width=1024,
                height=768
            )

        # ---- 7. 转换到机器人坐标系 ----
        T_rack510_in_robot, T_flange_in_robot, T_cam_in_flange = \
            self.compute_T_rack510_in_robot(T_tcp_in_robot, T_rack_in_cam=T_rack510_in_cam)

        Rack510DetectionTuple = namedtuple(
            "Rack510DetectionTuple",
            ["T_rack510_in_cam",
             "T_rack510_in_robot",
             "pred_results",
             "confidence",
             "label_name",
             "T_flange_in_robot",
             "T_cam_in_flange",
             "template_cloud_rack510"]
        )

        return Rack510DetectionTuple(
            T_rack510_in_cam,
            T_rack510_in_robot,
            pred_results,
            conf,
            label_name,
            T_flange_in_robot,
            T_cam_in_flange,
            self.template_cloud_rack510
        )


class TubeProcessor:
    def __init__(self,
                 rack_shape=(5, 10),
                 center_holes=[(2, 4), (2, 5)],
                 tube_spacing=0.025,
                 tube_diameter=0.021,
                 intrinsics=None):
        """
        :param rack_shape: 试管架形状 (行, 列)
        """

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

        self.rack_shape = rack_shape
        self.center_holes = center_holes
        self.tube_spacing = tube_spacing
        self.tube_diameter = tube_diameter

        self.intrinsics = intrinsics or {
            "fx": 606.627,
            "fy": 606.657,
            "cx": 324.281,
            "cy": 241.149
        }

    def process_tube(self, color_img,
                     depth_image,
                     pred_results,
                     template_cloud_rack510,
                     T_rack510_in_robot,
                     T_rack510_in_cam,
                     T_flange_in_robot,
                     T_cam_in_flange,
                     save_dirs,
                     visualize=False):
        """
        完整试管架 + 试管处理流水线
        """
        fx, fy, cx, cy = (self.intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        print("[DEBUG] 调用 extract_tube_point_clouds() ...")
        tube_data_list = extract_tube_point_clouds(color_img, depth_image, pred_results,
                                                   ['TubeB', 'TubeG', 'TubeO'],
                                                   fx, fy, cx, cy,
                                                   save_dirs["point_clouds"],
                                                   save_dirs["masks"],
                                                   save_dirs["planes"],
                                                   save_dirs["remaining"])
        print(f"[DEBUG] tube_data_list 数量: {len(tube_data_list)}")

        rack_pcd = template_cloud_rack510
        print(f"试管架点云点数: {len(rack_pcd.points)}")

        # 变换矩阵
        transformation_matrix = T_rack510_in_cam
        # 应用变换矩阵
        rack_pcd.transform(transformation_matrix)

        # 创建估计器
        estimator = SimplifiedTubePoseEstimator(rack_shape=self.rack_shape,
                                                center_holes=self.center_holes,
                                                tube_spacing=self.tube_spacing,
                                                tube_diameter=self.tube_diameter, )

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
        # 可视化
        # -------------------------
        if visualize:
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

            save_path = os.path.join(save_dirs["pcd"], "multiple_tubes_projection_result_rack34.png")

            # 可视化结果
            estimator.visualize_multiple_tubes(tube_results, label_colors=label_colors,
                                               save_path=save_path)

        # =============================================================================================

        # 试管位姿计算
        tubes_robot_info = []
        for result in tube_results:
            tube_center_cam = np.append(result.tube_points_raw_center, 1.0)

            # 转换到机器人坐标系
            tube_center_robot_h = T_flange_in_robot @ T_cam_in_flange @ tube_center_cam
            tube_center_robot = tube_center_robot_h[:3]

            rack_rot_robot = np.array(T_rack510_in_robot)[:3, :3]  # 取旋转部分
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

        rack_matrix = [[0 for _ in range(self.rack_shape[1])] for _ in range(self.rack_shape[0])]

        for result in tube_results:
            row, col = result.grid_position
            num_label = self.LABEL_TO_NUM.get(result.label, 0)  # 如果找不到对应的试管标签则设为 0
            rack_matrix[row][col] = num_label

        return rack_matrix, tubes_robot_info


class Rack510Pipeline:
    def __init__(self, weights_path="best_rack510.pt",
                 rack_shape=(5, 10),
                 center_holes=[(2, 4), (2, 5)],
                 tube_spacing=0.025,
                 tube_diameter=0.021
                 ):
        self.detector = Rack510Detector(weights_path=weights_path)
        self.tube_processor = TubeProcessor(rack_shape=rack_shape,
                                            center_holes=center_holes,
                                            tube_spacing=tube_spacing,
                                            tube_diameter=tube_diameter
                                            )
        self.result = None  # 包括试管架和试管的全部的位姿信息

    def _save_rack_and_tubes_info(self, save_folder, rack_results, rack_matrix, tubes_results):
        """
        保存 rack510 及试管信息
        :param save_folder: 保存路径
        :param result: RackAndTubesResult 数据类实例
        """
        if not rack_results and tubes_results:
            print("[WARN] 没有可保存的数据")
            return None

        robot_poses_data = {
            "rack_info": {
                "rack_label": rack_results.label_name,
                "rack_confidence": float(rack_results.confidence),
                "pose_in_camera": rack_results.T_rack510_in_cam.tolist(),
                "pose_in_robot": rack_results.T_rack510_in_robot.tolist(),
                "rack_matrix": rack_matrix
            },
            "tubes_info": tubes_results
        }

        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "rack_and_tubes_pose_robot.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)
        print(f"已保存到: {save_path}")

        return robot_poses_data

    def run(self, color_image, depth_image, T_tcp_in_robot, save_dirs):
        print("[INFO] === Rack510Pipeline.run() 开始执行 ===")
        # print(f"[INFO] 保存路径参数: {save_dirs}")
        print("[INFO] Step 1: 开始检测 rack510 ...")
        rack_result = self.detector.process_rack510_from_image(
            color_img=color_image,
            depth_img=depth_image,
            T_tcp_in_robot=T_tcp_in_robot,
            mask_dir=save_dirs["masks"],
            pointcloud_dir=save_dirs["point_clouds"],
            plane_dir=save_dirs["planes"],
            remaining_dir=save_dirs["remaining"]
        )
        print("[INFO] Step 1: rack510 检测完成")

        if rack_result is None:
            return None

        print("[INFO] Step 2: 开始处理试管（process_tube） ...")
        rack_matrix, tubes_robot_info = self.tube_processor.process_tube(color_image,
                                                                         depth_image,
                                                                         rack_result.pred_results,
                                                                         rack_result.template_cloud_rack510,
                                                                         rack_result.T_rack510_in_robot,
                                                                         rack_result.T_rack510_in_cam,
                                                                         rack_result.T_flange_in_robot,
                                                                         rack_result.T_cam_in_flange,
                                                                         save_dirs,
                                                                         visualize=True
                                                                         )
        print("[INFO] Step 2: 试管处理完成")

        print("[INFO] Step 3: 保存 rack 与 tubes 信息 ...")
        Rack510_and_tubes_result = self._save_rack_and_tubes_info(save_dirs["pcd"], rack_result, rack_matrix,
                                                                  tubes_robot_info)
        print("[INFO] Step 3: 数据保存完成")
        print("[INFO] === Rack510Pipeline.run() 执行结束 ===")

        return Rack510_and_tubes_result


class Rack34Detector:
    DEFAULT_WEIGHTS_PATH = "best_rack34.pt"

    def __init__(self,
                 weights_path=None,
                 sam_checkpoint=r"D:\AI\WRS-6DoF-PE\weights\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 use_new_intrinsics=False,
                 template_dir=None):

        """初始化固定资源"""
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.intrinsics = CAMERA_INTRINSICS_NEW if use_new_intrinsics else CAMERA_INTRINSICS
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH

        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"
        self.template_cloud_rack34 = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack34_plane.ply"))
        self.template_cloud_rack34_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "rack34.ply"))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_rack34_pose_in_robot(self,
                                     T_tcp_in_robot,
                                     T_cam_in_flange=None,
                                     T_tcp_offset=None,
                                     T_rack34_in_cam=None):
        """
        计算 已分检试管架 在机器人坐标系下的位姿

        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_rack34_in_cam: (4x4) 已分检试管架 在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) T_rack34_in_robot
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

        if T_rack34_in_cam is None:
            raise ValueError("必须传入 T_rack34_in_cam (试管架在相机坐标系下的位姿)")

        # 最终 T_rack34_in_robot
        T_rack34_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_rack34_in_cam
        print("[DEBUG] T_rack34_in_robot:\n", T_rack34_in_robot)

        return T_rack34_in_robot, T_flange_in_robot, T_cam_in_flange

    def process_rack34_from_image(self,
                                  color_img, depth_img,
                                  T_tcp_in_robot,
                                  mask_dir, pointcloud_dir,
                                  plane_dir, remaining_dir,
                                  visualize=True):

        """
        从 RGB + 深度图像中检测并配准 rack34,返回 T_rack34_in_cam

        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred_results: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: T_rack34_in_cam (4x4 numpy array) 或 None
        """

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(plane_dir, exist_ok=True)
        os.makedirs(remaining_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # 保存 img_detected
        img_detected_path = os.path.join(pointcloud_dir, 'img_detected.jpg')
        cv2.imwrite(img_detected_path, img_detected)
        print(f"img_detected 已保存: {img_detected_path}")

        # ---- 1. 找 rack34 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results, self.detect_api,
                                                                                         "rack34",
                                                                                         check_rotate=True)
        if largest_rack is None:
            print("[ERROR] 未检测到 rack34")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(
            point_cloud_path,
            plane_dir, remaining_dir,
            0.002, 20000
        )

        # ---- 5. ICP 位姿估计 ----
        T_rack34_in_cam = icp_plane_estimate_pose(plane_cloud, self.template_cloud_rack34)
        print("试管架在相机坐标系下的位姿：\n", T_rack34_in_cam)
        if T_rack34_in_cam is None:
            print("[ERROR] ICP 位姿估计失败")
            return None

            # ---- 6. 可视化（可选） ----
        if visualize:
            template_cloud_rack34_full = copy.deepcopy(self.template_cloud_rack34_full)
            aligned_scene_pcd = template_cloud_rack34_full.transform(T_rack34_in_cam)

            # 给颜色区分
            point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
            aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

            o3d.visualization.draw_geometries(
                [aligned_scene_pcd, point_cloud],
                window_name="After ICP Registration",
                width=1024,
                height=768
            )

        # ---- 7. 转换到机器人坐标系 ----
        T_rack34_in_robot, T_flange_in_robot, T_cam_in_flange = \
            self.compute_rack34_pose_in_robot(T_tcp_in_robot, T_rack34_in_cam=T_rack34_in_cam)

        Rack34DetectionTuple = namedtuple(
            "Rack34DetectionTuple",
            ["T_rack34_in_cam",
             "T_rack34_in_robot",
             "pred_results",
             "confidence",
             "label_name",
             "T_flange_in_robot",
             "T_cam_in_flange",
             "template_cloud_rack34"]
        )

        return Rack34DetectionTuple(
            T_rack34_in_cam,
            T_rack34_in_robot,
            pred_results,
            conf,
            label_name,
            T_flange_in_robot,
            T_cam_in_flange,
            self.template_cloud_rack34
        )


class Tube34Processor:
    def __init__(self,
                 rack_shape=(3, 4),
                 center_holes=[(1, 1), (1, 2)],
                 tube_spacing=0.03,
                 tube_diameter=0.021,
                 intrinsics=None,
                 ):
        """
        :param rack_shape: 试管架形状 (行, 列)
        """
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
        self.rack_shape = rack_shape
        self.center_holes = center_holes
        self.tube_spacing = tube_spacing
        self.tube_diameter = tube_diameter

        self.intrinsics = intrinsics or {
            "fx": 606.627,
            "fy": 606.657,
            "cx": 324.281,
            "cy": 241.149
        }

    def process_tube(self, color_img,
                     depth_image,
                     pred_results,
                     template_cloud_rack34,
                     T_rack34_in_robot,
                     T_rack34_in_cam,
                     T_flange_in_robot,
                     T_cam_in_flange,
                     save_dirs,
                     visualize=False):
        """
        完整试管架 + 试管处理流水线
        """
        fx, fy, cx, cy = (self.intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        print("[DEBUG] 调用 extract_tube_point_clouds() ...")
        tube_data_list = extract_tube_point_clouds(color_img, depth_image, pred_results,
                                                   ['TubeB', 'TubeG', 'TubeO'],
                                                   fx, fy, cx, cy,
                                                   save_dirs["point_clouds"],
                                                   save_dirs["masks"],
                                                   save_dirs["planes"],
                                                   save_dirs["remaining"])
        print(f"[DEBUG] tube_data_list 数量: {len(tube_data_list)}")

        rack_pcd = template_cloud_rack34
        print(f"试管架点云点数: {len(rack_pcd.points)}")

        # 变换矩阵
        transformation_matrix = T_rack34_in_cam
        # 应用变换矩阵
        rack_pcd.transform(transformation_matrix)

        # 创建估计器
        estimator = SimplifiedTubePoseEstimator(rack_shape=self.rack_shape,
                                                center_holes=self.center_holes,
                                                tube_spacing=self.tube_spacing,
                                                tube_diameter=self.tube_diameter,
                                                )

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
        # 可视化
        # -------------------------
        if visualize:
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

            save_path = os.path.join(save_dirs["pcd"], "multiple_tubes_projection_result_rack510.png")

            # 可视化结果
            estimator.visualize_multiple_tubes(tube_results, label_colors=label_colors, save_path=save_path)

            # # 可视化结果
            # estimator.visualize_multiple_tubes(tube_results, label_colors=label_colors,
            #                                    save_path="multiple_tubes_projection_result.png")

        # =============================================================================================

        # 试管位姿计算
        tubes_robot_info = []
        for result in tube_results:
            tube_center_cam = np.append(result.tube_points_raw_center, 1.0)

            # 转换到机器人坐标系
            tube_center_robot_h = T_flange_in_robot @ T_cam_in_flange @ tube_center_cam
            tube_center_robot = tube_center_robot_h[:3]

            rack_rot_robot = np.array(T_rack34_in_robot)[:3, :3]  # 取旋转部分
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

        rack_matrix = [[0 for _ in range(self.rack_shape[1])] for _ in range(self.rack_shape[0])]

        for result in tube_results:
            row, col = result.grid_position
            num_label = self.LABEL_TO_NUM.get(result.label, 0)  # 如果找不到对应的试管标签则设为 0
            rack_matrix[row][col] = num_label

        return rack_matrix, tubes_robot_info


class Rack34Pipeline:
    def __init__(self, weights_path="best_rack34.pt",
                 rack_shape=(3, 4),
                 center_holes=[(1, 1), (1, 2)],
                 tube_spacing=0.03,
                 tube_diameter=0.021):
        self.detector = Rack34Detector(weights_path=weights_path)
        self.tube_processor = Tube34Processor(rack_shape=rack_shape,
                                              center_holes=center_holes,
                                              tube_spacing=tube_spacing,
                                              tube_diameter=tube_diameter
                                              )
        self.result = None  # 包括试管架和试管的全部的位姿信息

    def _save_rack34_and_tubes_info(self, save_folder, rack_results, rack_matrix, tubes_results):
        """
        保存 rack510 及试管信息
        :param save_folder: 保存路径
        :param result: RackAndTubesResult 数据类实例
        """
        if not rack_results and tubes_results:
            print("[WARN] 没有可保存的数据")
            return None

        robot_poses_data = {
            "rack_info": {
                "rack_label": rack_results.label_name,
                "rack_confidence": float(rack_results.confidence),
                "pose_in_camera": rack_results.T_rack34_in_cam.tolist(),
                "pose_in_robot": rack_results.T_rack34_in_robot.tolist(),
                "rack_matrix": rack_matrix
            },
            "tubes_info": tubes_results
        }

        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "T_rack34_in_robot.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)
        print(f"已保存到: {save_path}")

        return robot_poses_data

    def run(self, color_image, depth_image, T_tcp_in_robot, save_dirs):
        print("[INFO] === Rack34Pipeline.run() 开始执行 ===")

        print("[INFO] Step 1: 开始检测 rack34 ...")
        rack_result = self.detector.process_rack34_from_image(
            color_img=color_image,
            depth_img=depth_image,
            T_tcp_in_robot=T_tcp_in_robot,
            mask_dir=save_dirs["masks"],
            pointcloud_dir=save_dirs["point_clouds"],
            plane_dir=save_dirs["planes"],
            remaining_dir=save_dirs["remaining"]
        )
        print("[INFO] Step 1: rack34 检测完成")

        if rack_result is None:
            return None

        print("[INFO] Step 2: 开始处理试管（process_tube） ...")
        rack_matrix, tubes_robot_info = self.tube_processor.process_tube(color_image,
                                                                         depth_image,
                                                                         rack_result.pred_results,
                                                                         rack_result.template_cloud_rack34,
                                                                         rack_result.T_rack34_in_robot,
                                                                         rack_result.T_rack34_in_cam,
                                                                         rack_result.T_flange_in_robot,
                                                                         rack_result.T_cam_in_flange,
                                                                         save_dirs,
                                                                         visualize=True
                                                                         )
        print("[INFO] Step 2: 试管处理完成")

        print("[INFO] Step 3: 保存 rack 与 tubes 信息 ...")
        Rack34_and_tubes_result = self._save_rack34_and_tubes_info(save_dirs["pcd"], rack_result, rack_matrix,
                                                                   tubes_robot_info)
        print("[INFO] Step 3: 数据保存完成")
        print("[INFO] === Rack510Pipeline.run() 执行结束 ===")

        return Rack34_and_tubes_result


class RefrigeratorDetector:
    DEFAULT_WEIGHTS_PATH = "best.pt"

    def __init__(self,
                 mode="open",  # "open" 或 "close"
                 weights_path=None,
                 sam_checkpoint=r"D:\AI\WRS-6DoF-PE\weights\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 use_new_intrinsics=False,
                 template_dir=None):
        """初始化固定资源"""
        self.mode = mode.lower()
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.intrinsics = CAMERA_INTRINSICS_NEW if use_new_intrinsics else CAMERA_INTRINSICS
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH

        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"
        if self.mode == "open":
            plane_file = "refrigerator_open_plane.ply"
            full_file = "refrigerator_open.ply"
        elif self.mode == "close":
            plane_file = "refrigerator_close_plane.ply"
            full_file = "refrigerator_close.ply"
        else:
            raise ValueError("mode 必须是 'open' 或 'close'")

        self.template_cloud_refrigerator = o3d.io.read_point_cloud(os.path.join(base_obj_dir, plane_file))
        self.template_cloud_refrigerator_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, full_file))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_T_refrigerator_in_robot(self,
                                        T_tcp_in_robot,
                                        T_cam_in_flange=None,
                                        T_tcp_offset=None,
                                        T_refrigerator_in_cam=None):

        """
        计算试管架在机器人坐标系下的位姿

        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_refrigerator_in_cam: (4x4) 冰箱 在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) T_refrigerator_in_robot

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
            T_tcp_offset = np.array([0, 0, 0.21285675])

            # 已知 TCP 相对于法兰的位姿 (固定)
        T_tcp_in_flange = np.eye(4)
        T_tcp_in_flange[:3, :3] = np.eye(3)
        T_tcp_in_flange[:3, 3] = T_tcp_offset

        # flange 相对于 robot 的位姿
        T_flange_in_robot = T_tcp_in_robot @ np.linalg.inv(T_tcp_in_flange)
        print("[DEBUG] T_flange_in_robot:\n", T_flange_in_robot)

        if T_refrigerator_in_cam is None:
            raise ValueError("必须传入 T_rack_in_cam (试管架在相机坐标系下的位姿)")

        # 最终 T_refrigerator_in_robot
        T_refrigerator_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_refrigerator_in_cam
        print("[DEBUG] T_refrigerator_in_robot:\n", T_refrigerator_in_robot)

        return T_refrigerator_in_robot, T_flange_in_robot, T_cam_in_flange

    def save_detection_result(self, result, save_folder):
        """
        保存 RefrigeratorDetectionTuple 结果到 JSON 文件
        :param result: RefrigeratorDetectionTuple
        :param save_folder: 保存目录
        """
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "T_refrigerator_in_robot_data.json")

        # 将 numpy 矩阵转为 list
        robot_poses_data = {
            "mode": self.mode,
            "label_name": result.label_name,
            "confidence": float(result.confidence),
            "T_refrigerator_in_cam": result.T_refrigerator_in_cam.tolist(),
            "T_refrigerator_in_robot": result.T_refrigerator_in_robot.tolist(),
            "T_flange_in_robot": result.T_flange_in_robot.tolist(),
            "T_cam_in_flange": result.T_cam_in_flange.tolist(),
            "pred_results": str(result.pred_results)
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)
        print(f"已保存到: {save_path}")

    def process_refrigerator_from_image(self,
                                        color_img, depth_img,
                                        T_tcp_in_robot,
                                        mask_dir, pointcloud_dir,
                                        plane_dir, remaining_dir,
                                        visualize=True
                                        ):

        """
        从 RGB + 深度图像中检测并配准 refrigerator,返回 T_refrigerator_in_cam

        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred_results: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: T_refrigerator_in_cam (4x4 numpy array) 或 None
        """

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(plane_dir, exist_ok=True)
        os.makedirs(remaining_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # ---- 1. 找 rack510 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results, self.detect_api,
                                                                                         "IC")
        if largest_rack is None:
            print("[ERROR] 未检测到 refrigerator")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(
            point_cloud_path,
            plane_dir, remaining_dir,
            0.002, 20000
        )

        # ---- 5. ICP 位姿估计 ----
        T_refrigerator_in_cam = icp_plane_estimate_pose(plane_cloud, self.template_cloud_refrigerator)
        print("试管架在相机坐标系下的位姿：\n", T_refrigerator_in_cam)
        if T_refrigerator_in_cam is None:
            print("[ERROR] ICP 位姿估计失败")
            return None

            # ---- 6. 可视化（可选） ----
        if visualize:
            template_cloud_refrigerator_full = copy.deepcopy(self.template_cloud_refrigerator_full)
            aligned_scene_pcd = template_cloud_refrigerator_full.transform(T_refrigerator_in_cam)

            # 给颜色区分
            point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
            aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

            o3d.visualization.draw_geometries(
                [aligned_scene_pcd, point_cloud],
                window_name="After ICP Registration",
                width=1024,
                height=768
            )

        # ---- 7. 转换到机器人坐标系 ----
        T_refrigerator_in_robot, T_flange_in_robot, T_cam_in_flange = \
            self.compute_T_refrigerator_in_robot(T_tcp_in_robot, T_refrigerator_in_cam=T_refrigerator_in_cam)

        RefrigeratorDetectionTuple = namedtuple(
            "RefrigeratorDetectionTuple",
            ["T_refrigerator_in_cam",
             "T_refrigerator_in_robot",
             "pred_results",
             "confidence",
             "label_name",
             "T_flange_in_robot",
             "T_cam_in_flange",
             "template_cloud_refrigerator"]
        )

        return RefrigeratorDetectionTuple(
            T_refrigerator_in_cam,
            T_refrigerator_in_robot,
            pred_results,
            conf,
            label_name,
            T_flange_in_robot,
            T_cam_in_flange,
            self.template_cloud_refrigerator
        )


class CentrifugeDetector:
    DEFAULT_WEIGHTS_PATH = "best.pt"

    def __init__(self,
                 mode="open",  # "open" 或 "close"
                 weights_path=None,
                 sam_checkpoint=r"D:\AI\WRS-6DoF-PE\weights\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 use_new_intrinsics=False,
                 template_dir=None):
        """初始化固定资源"""
        self.mode = mode.lower()
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.intrinsics = CAMERA_INTRINSICS_NEW if use_new_intrinsics else CAMERA_INTRINSICS
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH

        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"
        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"

        if self.mode == "open":
            plane_file = "centrifuge_open_plane.ply"
            full_file = "centrifuge_open.ply"
        elif self.mode == "close":
            plane_file = "centrifuge_close_plane.ply"
            full_file = "centrifuge_close.ply"
        else:
            raise ValueError("mode 必须是 'open' 或 'close'")
        self.template_cloud_centrifuge = o3d.io.read_point_cloud(os.path.join(base_obj_dir, plane_file))
        self.template_cloud_centrifuge_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, full_file))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_T_centrifuge_in_robot(self,
                                      T_tcp_in_robot,
                                      T_cam_in_flange=None,
                                      T_tcp_offset=None,
                                      T_centrifuge_in_cam=None):
        """
        计算 离心机 在机器人坐标系下的位姿

        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_centrifuge_in_cam: (4x4) 离心机在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) T_centrifuge_in_robot
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

        if T_centrifuge_in_cam is None:
            raise ValueError("必须传入 T_centrifuge_in_cam (离心机 在相机坐标系下的位姿)")

        # 最终 T_centrifuge_in_robot
        T_centrifuge_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_centrifuge_in_cam
        print("[DEBUG] T_centrifuge_in_robot:\n", T_centrifuge_in_robot)

        return T_centrifuge_in_robot, T_flange_in_robot, T_cam_in_flange

    def save_detection_result(self, result, save_folder):
        """
        保存 CentrifugeDetectionTuple 结果到 JSON 文件
        :param result: CentrifugeDetectionTuple
        :param save_folder: 保存目录
        """
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "T_centrifuge_in_robot.json")

        # 将 numpy 矩阵转为 list
        robot_poses_data = {
            "mode": self.mode,
            "label_name": result.label_name,
            "confidence": float(result.confidence),
            "T_centrifuge_in_cam": result.T_centrifuge_in_cam.tolist(),
            "T_centrifuge_in_robot": result.T_centrifuge_in_robot.tolist(),
            "T_flange_in_robot": result.T_flange_in_robot.tolist(),
            "T_cam_in_flange": result.T_cam_in_flange.tolist(),
            "pred_results": str(result.pred_results)
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)
        print(f"已保存到: {save_path}")

    def process_centrifuge_from_image(self,
                                      color_img, depth_img,
                                      T_tcp_in_robot,
                                      mask_dir, pointcloud_dir,
                                      plane_dir, remaining_dir,
                                      visualize=True):

        """
        从 RGB + 深度图像中检测并配准 centrifuge ,返回 T_centrifuge_in_cam

        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred_results: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: T_centrifuge_in_cam (4x4 numpy array) 或 None
        """

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(plane_dir, exist_ok=True)
        os.makedirs(remaining_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # ---- 1. 找 centrifuge 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results, self.detect_api,
                                                                                         "H1")
        if largest_rack is None:
            print("[ERROR] 未检测到 centrifuge")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(point_cloud_path,
                                                           plane_dir, remaining_dir,
                                                           0.002, 20000)

        # ---- 5. ICP 位姿估计 ----
        T_centrifuge_in_cam = icp_plane_estimate_pose(plane_cloud, self.template_cloud_centrifuge)
        print("离心机 在相机坐标系下的位姿：\n", T_centrifuge_in_cam)
        if T_centrifuge_in_cam is None:
            print("[ERROR] ICP 位姿估计失败")
            return None

        # ---- 6. 可视化（可选） ----
        if visualize:
            template_cloud_centrifuge_full = copy.deepcopy(self.template_cloud_centrifuge_full)
            aligned_scene_pcd = template_cloud_centrifuge_full.transform(T_centrifuge_in_cam)

            # 给颜色区分
            point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
            aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

            o3d.visualization.draw_geometries([aligned_scene_pcd, point_cloud],
                                              window_name="After ICP Registration",
                                              width=1024,
                                              height=768
                                              )

        # ---- 7. 转换到机器人坐标系 ----
        T_centrifuge_in_robot, T_flange_in_robot, T_cam_in_flange = \
            self.compute_T_centrifuge_in_robot(T_tcp_in_robot, T_centrifuge_in_cam=T_centrifuge_in_cam)

        CentrifugeDetectionTuple = namedtuple(
            "CentrifugeDetectionTuple",
            ["T_centrifuge_in_cam",
             "T_centrifuge_in_robot",
             "pred_results",
             "confidence",
             "label_name",
             "T_flange_in_robot",
             "T_cam_in_flange",
             "template_cloud_centrifuge"]
        )

        return CentrifugeDetectionTuple(
            T_centrifuge_in_cam,
            T_centrifuge_in_robot,
            pred_results,
            conf,
            label_name,
            T_flange_in_robot,
            T_cam_in_flange,
            self.template_cloud_centrifuge
        )


class LockerDetector:
    DEFAULT_WEIGHTS_PATH = "best.pt"

    def __init__(self,
                 weights_path=None,
                 sam_checkpoint=r"D:\AI\WRS-6DoF-PE\weights\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 use_new_intrinsics=False,
                 template_dir=None):
        """初始化固定资源"""
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.intrinsics = CAMERA_INTRINSICS_NEW if use_new_intrinsics else CAMERA_INTRINSICS
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH

        # 模板点云路径
        base_obj_dir = template_dir or r"D:\AI\WRS-6DoF-PE\robot_grasp\objects"
        self.template_cloud_locker = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "locker_plane.ply"))
        self.template_cloud_locker_full = o3d.io.read_point_cloud(os.path.join(base_obj_dir, "locker.ply"))

        # === 初始化 YOLO 检测器 ===
        self.detect_api = DetectAPI(weights=self.weights_path)

    def compute_T_locker_in_robot(self,
                                  T_tcp_in_robot,
                                  T_cam_in_flange=None,
                                  T_tcp_offset=None,
                                  T_locker_in_cam=None):
        """
        计算 储物柜 在机器人坐标系下的位姿

        :param T_tcp_in_robot: (4x4) 位姿矩阵（来自检测）
        :param T_cam_in_flange: (4x4) 相机在夹爪法兰坐标系下的位姿,默认使用标定好的固定值
        :param T_tcp_offset: (3,) TCP 相对于法兰的 Z 方向偏移
        :param T_locker_in_cam: (4x4) 储物柜 在相机坐标系下的位姿（来自位姿估计）
        :return: (4x4) T_locker_in_robot
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

        if T_locker_in_cam is None:
            raise ValueError("必须传入 T_locker_in_cam (试管架在相机坐标系下的位姿)")

        # 最终 T_locker_in_robot
        T_locker_in_robot = T_flange_in_robot @ T_cam_in_flange @ T_locker_in_cam
        print("[DEBUG] T_locker_in_robot:\n", T_locker_in_robot)

        return T_locker_in_robot, T_flange_in_robot, T_cam_in_flange

    def save_detection_result(self, result, save_folder):
        """
        保存 LockerDetectionTuple 结果到 JSON 文件
        :param result: LockerDetectionTuple
        :param save_folder: 保存目录
        """
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "T_locker_in_robot.json")

        # 将 numpy 矩阵转为 list
        robot_poses_data = {
            "label_name": result.label_name,
            "confidence": float(result.confidence),
            "T_locker_in_cam": result.T_locker_in_cam.tolist(),
            "T_locker_in_robot": result.T_locker_in_robot.tolist(),
            "T_flange_in_robot": result.T_flange_in_robot.tolist(),
            "T_cam_in_flange": result.T_cam_in_flange.tolist(),
            "pred_results": str(result.pred_results)
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(robot_poses_data, f, indent=4, ensure_ascii=False)
        print(f"已保存到: {save_path}")

    def process_locker_from_image(self,
                                  color_img, depth_img,
                                  T_tcp_in_robot,
                                  mask_dir, pointcloud_dir,
                                  plane_dir, remaining_dir,
                                  visualize=True):
        """
        从 RGB + 深度图像中检测并配准 locker,返回 T_locker_in_cam

        :param color_img: RGB 图 (numpy)
        :param depth_img: 深度图 (numpy)
        :param pred_results: 目标检测预测结果
        :param mask_dir: 保存 mask 的目录
        :param pointcloud_dir: 保存点云的目录
        :return: T_locker_in_cam (4x4 numpy array) 或 None
        """

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(plane_dir, exist_ok=True)
        os.makedirs(remaining_dir, exist_ok=True)

        img_detected, pred_results, _ = self.detect_api.detect([color_img])

        # ---- 1. 找 locker 最大检测框 ----
        largest_rack, rotate_matrix, bbox_rack, conf, label_name = select_largest_object(pred_results,
                                                                                         self.detect_api, "G0")
        if largest_rack is None:
            print("[ERROR] 未检测到 locker")
            return None

        # ---- 2. 分割 mask ----
        mask = sam_segment(color_img, bbox_rack)
        mask_image, _ = save_object_mask_on_black_background(color_img, mask, bbox_rack, label_name, mask_dir)

        # ---- 3. 生成点云 ----
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        point_cloud = extract_point_cloud(color_img, mask_image, depth_img, fx, fy, cx, cy)
        point_cloud_path = os.path.join(pointcloud_dir, f"point_cloud_{label_name}.ply")
        o3d.io.write_point_cloud(point_cloud_path, point_cloud)

        # ---- 4. 提取平面 ----
        plane_cloud, remaining_cloud, _, _ = extract_plane(point_cloud_path,
                                                           plane_dir, remaining_dir,
                                                           0.002, 20000
                                                           )

        # ---- 5. ICP 位姿估计 ----
        T_locker_in_cam = icp_plane_estimate_pose(plane_cloud, self.template_cloud_locker)
        print("试管架在相机坐标系下的位姿：\n", T_locker_in_cam)
        if T_locker_in_cam is None:
            print("[ERROR] ICP 位姿估计失败")
            return None

        # ---- 6. 可视化（可选） ----
        if visualize:
            template_cloud_locker_full = copy.deepcopy(self.template_cloud_locker_full)
            aligned_scene_pcd = template_cloud_locker_full.transform(T_locker_in_cam)

            # 给颜色区分
            point_cloud.paint_uniform_color([1, 0, 0])  # 红色：场景点云
            aligned_scene_pcd.paint_uniform_color([0, 1, 0])  # 绿色：对齐后模板

            o3d.visualization.draw_geometries(
                [aligned_scene_pcd, point_cloud],
                window_name="After ICP Registration",
                width=1024,
                height=768
            )

        # ---- 7. 转换到机器人坐标系 ----
        T_locker_in_robot, T_flange_in_robot, T_cam_in_flange = \
            self.compute_T_locker_in_robot(T_tcp_in_robot, T_locker_in_cam=T_locker_in_cam)

        LockerDetectionTuple = namedtuple(
            "LockerDetectionTuple",
            ["T_locker_in_cam",
             "T_locker_in_robot",
             "pred_results",
             "confidence",
             "label_name",
             "T_flange_in_robot",
             "T_cam_in_flange",
             "template_cloud_locker"]
        )

        return LockerDetectionTuple(
            T_locker_in_cam,
            T_locker_in_robot,
            pred_results,
            conf,
            label_name,
            T_flange_in_robot,
            T_cam_in_flange,
            self.template_cloud_locker
        )
