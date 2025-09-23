#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：ABB_wrs_hu_sam
@File    ：robot_grasp.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""
import os.path
import pickle
import numpy as np
import basis.robot_math as rm
from manipulation.pick_place_planner import PickPlacePlanner
from motion.probabilistic.rrt_connect import RRTConnect
from robot_sim.end_effectors.gripper.ag145.ag145 import Ag145
from robot_sim.robots.gofa5.gofa5_Ag145 import GOFA5
from grasping.planning import antipodal as gpa

current_dir = os.path.dirname(os.path.abspath(__file__))


def save_path_data(path_file, conf_list, jawwidth_list, objpose_list):
    """保存路径数据"""
    try:
        # 确保保存路径的目录存在
        save_dir = os.path.dirname(path_file)  # 获取文件所在的目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 如果目录不存在,则创建它

        with open(path_file, "wb") as f:
            pickle.dump({'conf_list': conf_list, 'jawwidth_list': jawwidth_list, 'objpose_list': objpose_list}, f)
        print(f"路径数据已成功保存到: {path_file}")  # 增加一个成功的日志
    except Exception as e:
        print(f"保存路径文件 {path_file} 时出错: {e}")


def load_path_data(path_file):
    """从文件加载路径数据

    Args:
        path_file (str): 文件名
    Returns:
        tuple: (conf_list, jawwidth_list, objpose_list),如果文件不存在,返回None
    """
    try:
        with open(path_file, "rb") as f:
            conf_list, jawwidth_list, objpose_list = pickle.load(f)
        print(f"成功加载路径文件,文件地址为:{path_file}")
        return conf_list, jawwidth_list, objpose_list
    except FileNotFoundError:
        print(f"未找到相应的路径文件:{path_file}")
        return None, None, None

class RobotGrasping:
    def __init__(self, base, rbt_s):
        self.base = base
        self.rbt_s = rbt_s  # 直接引用外部的实例
        self.rrtc_s = RRTConnect(self.rbt_s)
        self.ppp_s = PickPlacePlanner(self.rbt_s)
        self.manipulator_name = "arm"
        self.hand_name = "hnd"
        # 如果外部已经初始化过 gripper,不一定要创建新的 Ag145
        # self.gripper_s = Ag145()
        self.grasp_file_map = {
            "TubeB": ("blood_tube10_center", "ag145_grasps_blood_tube10_center.pickle"),
            "TubeG": ("blood_tube10_center", "ag145_grasps_blood_tube10_center.pickle"),
            "TubeO": ("blood_tube10_center", "ag145_grasps_blood_tube10_center.pickle")
        }
        self.current_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # self.rbt_r = gofa_con.GoFaArmController()  # 初始化机器人控制对象

    def load_grasp_info_by_tube_type(self, tube_type):
        """
        根据试管类型加载对应的抓取规范（pickle）
        Args:
            tube_type (str): TubeB / TubeG / TubeO 等
        Returns:
            list: 抓取信息列表
        """
        if tube_type not in self.grasp_file_map:
            raise ValueError(f"未知的试管类型: {tube_type},请检查 grasp_file_map 设置.")

        dataset_name, file_name = self.grasp_file_map[tube_type]
        return gpa.load_pickle_file(dataset_name, root=current_dir, file_name=file_name)


    def grasp_tube_invocation(self, tube, goal_pos, goal_rotmat, obstacle_list, tube_row, tube_col, current_jnts,
                              save_HRC_paths_dir):
        """抓取指定位置的试管
        Args:
            tube (CollisionModel): 要抓取的试管对象
            goal_pos (np.ndarray): 目标位置
            goal_rotmat (np.ndarray): 目标旋转
            obstacle_list (list): 障碍物列表
            tube_row (int): 试管在矩阵中的行索引
            tube_col (int): 试管在矩阵中的列索引
        Returns:
            tuple: (conf_list, jawwidth_list, objpose_list),如果规划失败,返回 None
        """
        # 定义保存路径的文件名,包含试管信息
        PATH_FILE = os.path.join(save_HRC_paths_dir,
                                 f"robot_motion_saved_paths\saved_path_{tube_row}_{tube_col}.pickle")

        # 检查是否存在保存的路径文件
        conf_list, jawwidth_list, objpose_list = load_path_data(PATH_FILE)

        # 如果没有加载到路径,则重新规划路径
        if not conf_list:
            # 确保要抓取的试管存在
            if tube is not None:
                print("当前夹爪宽度:", self.rbt_s.get_gripper_width())

                # ===== 调试信息：移除前的状态 =====
                print(f"\n=== 试管移除验证 (位置: {tube_row}, {tube_col}) ===")
                print(f"移除前障碍物列表长度: {len(obstacle_list)}")
                tube_in_list_before = tube in obstacle_list
                print(f"目标试管是否在障碍物列表中: {tube_in_list_before}")

                # 如果需要更详细的信息,可以打印试管的ID或位置
                if hasattr(tube, 'name'):
                    print(f"目标试管名称: {tube.name}")
                tube_pos = tube.get_pos()
                print(f"目标试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}]")

                # 获取并打印旋转矩阵
                tube_rotmat = tube.get_rotmat()
                print(f"目标试管旋转矩阵:\n{tube_rotmat}")

                # 可选：打印旋转矩阵的欧拉角表示 (roll, pitch, yaw)
                tube_rpy = rm.rotmat_to_euler(tube_rotmat)
                # Assuming 'rm' is your robotics math library (e.g., roboticstoolbox)
                print(f"目标试管欧拉角 (roll, pitch, yaw): [{tube_rpy[0]:.3f}, {tube_rpy[1]:.3f}, {tube_rpy[2]:.3f}]")

                # 移除要抓取的试管,避免运动规划时将其视为障碍物
                if tube in obstacle_list:
                    obstacle_list.remove(tube)
                    print(f"成功从障碍物列表中移除目标试管")
                else:
                    print(f"目标试管本来就不在障碍物列表中")

                # ===== 调试信息：移除后的状态 =====
                print(f"移除后障碍物列表长度: {len(obstacle_list)}")
                tube_in_list_after = tube in obstacle_list
                print(f"移除后目标试管是否还在障碍物列表中: {tube_in_list_after}")

                # 验证移除是否成功
                if tube_in_list_before and not tube_in_list_after:
                    print(f"验证成功：目标试管已被正确移除")
                elif not tube_in_list_before:
                    print(f"目标试管原本就不在障碍物列表中")
                else:
                    print(f"警告：目标试管移除失败！")
                print("=" * 50)

                start_pos = tube.get_pos()
                start_rotmat = tube.get_rotmat()
                obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
                obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

                # 假设 tube_obj 有属性 tube_type（或你可以从外部调用时传入）
                tube_type = getattr(tube, "tube_type", "TubeB")  # 默认 TubeB
                grasp_info_list = self.load_grasp_info_by_tube_type(tube_type)

                conf_list, jawwidth_list, objpose_list = \
                    self.ppp_s.gen_pick_and_place_motion(hnd_name=self.hand_name,
                                                         objcm=tube,
                                                         grasp_info_list=grasp_info_list,
                                                         start_conf=current_jnts,
                                                         end_conf=None,
                                                         goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                                         approach_direction_list=[None, np.array([0, 0, -1])],
                                                         approach_distance_list=[.2] * 2,
                                                         depart_direction_list=[np.array([0, 0, 1]), None],
                                                         depart_distance_list=[.2] * 2,
                                                         obstacle_list=obstacle_list,
                                                         approach_jawwidth=0.023/2,
                                                         depart_jawwidth=0.023/2,
                                                         )

                # 保存路径数据到文件
                if conf_list:
                    save_path_data(PATH_FILE, conf_list, jawwidth_list, objpose_list)
                    obstacle_list.append(tube)
                    print(f"成功保存位置 ({tube_row}, {tube_col}) 的路径数据！")

                    # 调用机器人控制对象执行运动
                    # self.rbt_r.move_jntspace_path(conf_list)

                    return conf_list, jawwidth_list, objpose_list
                else:
                    # 规划失败,将试管添加回 obstacle_list,以便后续的碰撞检测
                    obstacle_list.append(tube)
                    print(f"位置 ({tube_row}, {tube_col}) 的试管无法生成运动轨迹")
                    return None, None, None
            else:
                print(f"位置 ({tube_row}, {tube_col}) 的试管不存在,无法生成运动轨迹")
                return None, None, None
        else:
            print(f"成功加载位置 ({tube_row}, {tube_col}) 的路径数据！")
            # 调用机器人控制对象执行运动
            # self.rbt_r.move_jntspace_path(conf_list)

            return conf_list, jawwidth_list, objpose_list

    def move_to_configuration(self, conf):
        """移动机器人到指定的关节角度
        Args:
            conf (np.ndarray): 目标关节角度
        """
        self.rbt_s.fk(self.manipulator_name, conf)
        self.rbt_s.gen_meshmodel().attach_to(self.base)

    def detach_tube(self, tube):
        """从场景中移除试管
        Args:
            tube (CollisionModel): 要移除的试管对象
        """
        tube.detach()
