#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：WRS-6DoF-PE
@File    ：camera_utils.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""
import datetime
import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


def capture_point_cloud(save_dir):
    """采集对齐的彩色点云(含完整性检查) """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 1. 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置流(强制相同分辨率)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 2. 启动设备
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 获取深度单位比例

    # 3. 对齐工具
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # 4. 帧采集循环
        while True:
            # 等待一组连贯的帧
            frames = pipeline.wait_for_frames()

            # 防御性检查1:验证帧同步性
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if abs(depth_frame.timestamp - color_frame.timestamp) > 20:  # 1ms阈值
                print("[警告] 深度与彩色帧时间戳不同步！差值: {:.2f}ms".format(
                    abs(depth_frame.timestamp - color_frame.timestamp)))
                continue

            # 执行硬件对齐
            aligned_frames = align.process(frames)
            aligned_depth = aligned_frames.get_depth_frame()
            aligned_color = aligned_frames.get_color_frame()

            # 转换为numpy数组
            depth_image = np.asanyarray(aligned_depth.get_data())
            color_image = np.asanyarray(aligned_color.get_data())

            # 防御性检查2:可视化对齐效果
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            blended = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)
            cv2.imshow('Alignment Check', blended)

            # 等待用户按下回车键以捕获点云
            key = cv2.waitKey(1)
            if key == 13:  # 如果按下回车键,开始处理点云

                # 创建点云对象
                pc = rs.pointcloud()  # 创建RealSense点云对象
                pc.map_to(color_frame)  # 将彩色帧映射到点云中,用于提取纹理颜色
                points = pc.calculate(depth_frame)  # 基于深度图计算点云

                # 将点云的顶点(3D坐标)转换为Open3D格式
                vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # 获取点云顶点并转换为NumPy数组
                tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # 获取点云纹理坐标(UV)

                # 使用Open3D创建一个新的点云对象并加载顶点数据
                point_cloud = o3d.geometry.PointCloud()  # 创建Open3D点云对象
                point_cloud.points = o3d.utility.Vector3dVector(vtx)  # 设置点云的三维坐标数据

                # 修正UV坐标,确保UV坐标不超出[0, 1]的范围
                tex[:, 0] = np.clip(tex[:, 0], 0, 1)  # 限制U坐标在[0, 1]之间
                tex[:, 1] = np.clip(tex[:, 1], 0, 1)  # 限制V坐标在[0, 1]之间

                # 将UV坐标转换为图像的像素坐标
                tex_x = (tex[:, 0] * (color_image.shape[1] - 1)).astype(int)  # 将归一化的U坐标转换为像素横坐标
                tex_y = (tex[:, 1] * (color_image.shape[0] - 1)).astype(int)  # 将归一化的V坐标转换为像素纵坐标

                # 将彩色图像从BGR转换为RGB格式
                color_image_rgb = color_image[:, :, [2, 1, 0]]  # 将彩色图像的BGR格式转换为RGB格式

                # 根据UV坐标提取对应的RGB颜色信息
                colors = color_image_rgb[tex_y, tex_x, :]  # 使用像素坐标从RGB图像中提取颜色
                point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 将颜色数据归一化并赋值给点云

                # 生成文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # 保存RGB图
                filename_color = f"color_{timestamp}.png"
                save_colorpath = os.path.join(save_dir, filename_color)
                cv2.imwrite(save_colorpath, color_image)

                filename_depth = f"depth_{timestamp}.png"
                save_depthpath = os.path.join(save_dir, filename_depth)
                cv2.imwrite(save_depthpath, depth_image)

                # cv2.imwrite(os.path.join(save_dir, f"depth_{timestamp}.png"), depth_image)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f"pointcloud_{timestamp}.ply"), point_cloud)
                filename_pcd = f"pointcloud_{timestamp}.ply"
                save_pcdpath = os.path.join(save_dir, filename_pcd)

                if point_cloud is not None:
                    # 保存带颜色的点云到文件
                    o3d.io.write_point_cloud(save_pcdpath, point_cloud)  # 保存点云为PLY格式文件
                    print(f"颜色图已保存:{save_colorpath}")  # 输出保存成功信息
                    print(f"深度图已保存:{save_depthpath}")  # 输出保存成功信息
                    print(f"点云文件已保存:{save_pcdpath}")  # 输出保存成功信息
                    return color_image, save_colorpath, depth_image, save_depthpath, point_cloud, save_pcdpath,


                else:
                    return None, None, None, None, None, None  # 发生错误时返回 None

    except Exception as e:
        print(f"发生错误: {e}")
        return None, None, None, None, None, None  # 发生错误时返回 None

    finally:
        # 停止相机流并关闭OpenCV窗口
        pipeline.stop()  # 停止相机捕获
        cv2.destroyAllWindows()  # 关闭OpenCV窗口


if __name__ == "__main__":
    save_dir = r"D:\AI\WRS-6DoF-PE\robot_grasp\data"
    os.makedirs(save_dir, exist_ok=True)
    color_image, save_colorpath, depth_image, save_depthpath, point_cloud, save_pcdpath = capture_point_cloud(
        save_dir)  # 调用主函数,开始捕获点云
    if point_cloud is not None:
        print("点云获取成功！")
    else:
        print("点云获取失败！")
