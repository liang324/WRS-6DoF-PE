#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：WRS-6DoF-PE
@File    ：robotgrasp_run.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""

from robot_grasp.local_detection_utils import RobotController, execute_robot_with_command, \
    setup_kinectv3_camera_params, kinectv3_capture_and_detect
import sys

if __name__ == '__main__':
    # 初始化控制器
    robot_ctrl = RobotController()

    # 执行机械臂动作（人工指令）
    init_done00, place_done01 = execute_robot_with_command(robot_ctrl)
    # robot_ctrl.shutdown()

    if init_done00 and place_done01:
        print("[INFO] 进入拍摄流程...")

        # 设置相机参数
        depth_intrin, T_camera_to_robot, depth_scale, Kinect_data_savefolder = setup_kinectv3_camera_params()

        results, img_color = kinectv3_capture_and_detect(
            Kinect_data_savefolder, depth_intrin, T_camera_to_robot, depth_scale)

        # 判断检测结果
        if results and len(results) > 0:
            print(f"[INFO] 检测成功,检测到 {len(results)} 个目标,进入后续任务...")

            # 读取配置
            object_model_map, object_color_map = robot_ctrl.get_object_configs()

            # 构建障碍物
            obstacles = robot_ctrl.build_obstacles(results, object_model_map, object_color_map)

            # 自动移动执行
            robot_ctrl.auto_move_all_objects(results, move_real_flag=True, obstacles=obstacles)

        else:
            print("[ERROR] 目标检测失败或未检测到目标,跳过后续任务！")
            # 可选：结束任务或重新拍摄
            retry = input("是否重新拍摄并检测？(y/n): ").strip().lower()
            if retry == "y":
                results, img_color = kinectv3_capture_and_detect(
                    Kinect_data_savefolder, depth_intrin, T_camera_to_robot, depth_scale
                )
            else:
                sys.exit(1)

    else:
        print("[ERROR] 机械臂步骤未完成,禁止拍摄")
