"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: Capture_images_D435_optimization.py
Description:

"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from PIL import Image

# 定义保存图像的目录
save_directory = r'D:\AI\WRS-6DoF-PE\robot_grasp\data'

# 确保目录存在
os.makedirs(save_directory, exist_ok=True)

# 初始化管道
pipeline = rs.pipeline()

config = rs.config()

# 配置相机流
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 90)
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

# 设置对齐器,将深度图与彩色图像对齐
align_to = rs.stream.color  # 对齐到颜色流
align = rs.align(align_to)

try:
    while True:
        # 获取一帧数据
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = align.process(frames)

        # 获取对齐后的彩色帧和深度帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # # 获取彩色图像的内参
        # color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        # print(f"Camera Intrinsics:")
        # print(f"Width: {color_intrinsics.width}, Height: {color_intrinsics.height}")
        # print(f"fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
        # print(f"ppx: {color_intrinsics.ppx}, ppy: {color_intrinsics.ppy}")lllll
        # print(f"Distortion Model: {color_intrinsics.model}")
        # print(f"Distortion Coefficients: {color_intrinsics.coeffs}")

        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 显示彩色图像
        cv2.imshow('Color Image', color_image)

        # 按 Enter 键保存RGB图像和深度图像
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # # 保存RGB图像
            # color_image_path = os.path.join(save_directory, f'color_image_{timestamp}.jpg')
            # cv2.imwrite(color_image_path, color_image)
            # print(f'Saved color image as {color_image_path}')

            # 保存RGB图像
            color_image_path = os.path.join(save_directory, f'color_image_{timestamp}.png')
            cv2.imwrite(color_image_path, color_image)
            print(f'Saved color image as {color_image_path}')

            # # 保存深度图像
            # depth_image_path = os.path.join(save_directory, f'depth_image_{timestamp}.png')
            # # cv2.imwrite(depth_image_path, depth_image)
            # cv2.imwrite(depth_image_path, depth_image.astype(np.uint16))
            # print(f'Saved depth image as {depth_image_path}')

            # 保存深度图像 (使用 Pillow)
            depth_image_path = os.path.join(save_directory, f'depth_image_{timestamp}.png')
            depth_image = depth_image.astype(np.uint16)  # 确保深度图像是 uint16 类型
            img = Image.fromarray(depth_image)
            img.save(depth_image_path)
            print(f'Saved depth image as {depth_image_path}')

        # 按 'q' 键退出
        if key & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()