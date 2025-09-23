import pickle
import time
from typing import Literal
import multiprocessing as mp

import numpy as np
import pyrealsense2 as rs
import cv2

aruco = cv2.aruco
try:
    import cv2

    aruco = cv2.aruco
except:
    print("Cv2 aruco不存在,一些功能会停止")

__VERSION__ = '0.0.2'

# 阅读数据手册的第4章以获取详细信息
DEPTH_RESOLUTION_MID = (848, 480)
COLOR_RESOLUTION_MID = (848, 480)
DEPTH_RESOLUTION_HIGH = (1280, 720)
COLOR_RESOLUTION_HIGH = (1280, 720)
DEPTH_FPS = 30
COLOR_FPS = 30


def find_devices():
    '''
    查找连接到计算机的Realsense设备

    :return: 返回设备序列号列表和上下文
    '''
    ctx = rs.context()  # 创建librealsense上下文以管理设备
    serials = []
    if (len(ctx.devices) > 0):
        for dev in ctx.devices:
            print('Found device: ', dev.get_info(rs.camera_info.name), ' ', dev.get_info(rs.camera_info.serial_number))
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("没有连接的Intel设备")

    return serials, ctx


def stream_data(pipe: rs.pipeline, pc: rs.pointcloud) -> (np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray):
    '''
    为 RealSense流数据

    :param pipe: rs.pipeline 管道
    :param pc: rs.pointcloud 点云
    :return: 点云、点云颜色、深度图像和颜色图像
    '''
    # 获取一帧
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # 获取深度和颜色图像
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 计算点云和点云的颜色纹理
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # 计算点云的归一化颜色 (rgb nx3)
    cw, ch = color_image.shape[:2][::-1]
    v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)
    pc_color = color_image[u, v] / 255
    pc_color[:, [0, 2]] = pc_color[:, [2, 0]]
    return (verts, pc_color, depth_image, color_image)


class RealSenseD405(object):
    def __init__(self, resolution: Literal['mid', 'high'] = 'mid', device: str = None):
        """
        初始化RealSense D405设备

        :param resolution: 分辨率选择,可以是'mid'或'high'
        :param device: 设备序列号(可选)
        """
        assert resolution in ['mid', 'high']
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if device is not None:
            self._config.enable_device(device)

        # 设置配置
        if resolution == 'high':
            depth_resolution = DEPTH_RESOLUTION_HIGH
            color_resolution = COLOR_RESOLUTION_HIGH
        else:
            depth_resolution = DEPTH_RESOLUTION_MID
            color_resolution = COLOR_RESOLUTION_MID

        self._config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16,
                                   DEPTH_FPS)  # 启用深度流
        self._config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8,
                                   COLOR_FPS)  # 启用颜色流
        # 使用选择的配置开始流
        self._profile = self._pipeline.start(self._config)
        # 声明点云对象,用于计算点云和纹理映射
        self._pc = rs.pointcloud()

        color_frame = self._pipeline.wait_for_frames().get_color_frame()
        self._color_intr = color_frame.profile.as_video_stream_profile().intrinsics
        self.intr_mat = np.array([[self._color_intr.fx, 0, self._color_intr.ppx],
                                  [0, self._color_intr.fy, self._color_intr.ppy],
                                  [0, 0, 1]])
        self.intr_distcoeffs = np.asarray(self._color_intr.coeffs)

    def req_data(self):
        """
        请求数据:1) 点云,2) 点云颜色,3) 深度图像,4) 颜色图像

        :return: List[np.array, np.array, np.array, np.array]
        """
        return stream_data(pipe=self._pipeline, pc=self._pc)

    def get_pcd(self, return_color=False):
        """
        获取点云数据.如果return_color为True,还返回点云颜色

        :return: nx3 np.array
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        if return_color:
            return pcd, pcd_color
        return pcd

    def get_color_img(self):
        """
        获取颜色图像

        :return: 颜色图像
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        return color_img

    def get_depth_img(self):
        """
        获取深度图像

        :return: 深度图像
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        return depth_img

    def get_pcd_texture_depth(self):
        """
        返回点云、点云颜色、深度图像和颜色图像

        :return: List[np.array, np.array, np.array, np.array]
        """
        return self.req_data()

    def stop(self):
        '''
        停止以太网通信的子进程,允许程序优雅退出
        '''
        self._pipeline.stop()

    def recognize_ar_marker(self, aruco_dict=aruco.DICT_4X4_250, aruco_marker_size=.02, toggle_show=False):
        '''
        识别AR标记的功能

        :param aruco_dict: AR标记字典
        :param aruco_marker_size: AR标记大小
        :param toggle_show: 是否显示识别结果
        :return: 识别到的AR标记信息
        :return:
        '''
        color_img = self.get_color_img()
        parameters = aruco.DetectorParameters_create()
        aruco_dict = aruco.Dictionary_get(aruco_dict)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(color_img, aruco_dict, parameters=parameters,
                                                              cameraMatrix=self.intr_mat,
                                                              distCoeff=self.intr_distcoeffs)
        poselist = []
        detected_r = {}
        if ids is not None:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, self.intr_mat,
                                                                       self.intr_distcoeffs)
            if toggle_show:
                color_img = aruco.drawDetectedMarkers(color_img, corners, ids)
            for i in range(ids.size):
                rot = cv2.Rodrigues(rvecs[i])[0]
                pos = tvecs[i][0].ravel()
                homomat = np.eye(4)
                homomat[:3, :3] = rot
                homomat[:3, 3] = pos
                poselist.append(homomat)

                if toggle_show:
                    color_img = aruco.drawAxis(color_img, self.intr_mat, self.intr_distcoeffs, rvecs[i],
                                               tvecs[i], 0.03)
                #    aruco.drawAxis()
        if ids is None:
            idslist = []
        else:
            idslist = ids.ravel().tolist()
        if len(idslist) > 0:
            for ind, key in enumerate(idslist):
                detected_r[key] = poselist[ind]
        if toggle_show:
            return color_img, detected_r
        else:
            return detected_r

    def __del__(self):
        self.stop()

    def reset(self):
        """
        重置设备
        """
        device = self._profile.get_device()
        device.hardware_reset()
        del self


if __name__ == "__main__":
    import cv2
    import datetime

    # import huri.vision.yolov6.detect as yyd
    # from huri.core.common_import import fs

    serials, ctx = find_devices()
    print(serials)
    rs_pipelines = []
    for ser in serials:
        # 为每个设备序列号创建RealSenseD405对象
        rs_pipelines.append(RealSenseD405(device=ser))
        rs_pipelines[-1].reset()  # 重置设备
        time.sleep(5)  # 等待5秒
        rs_pipelines[-1] = RealSenseD405(device=ser)  # 重新初始化设备
        print("?")

    while True:
        for ind, pipeline in enumerate(rs_pipelines):
            img, results = pipeline.recognize_ar_marker(aruco_marker_size=0.045, toggle_show=True)
            cv2.imshow(f"color image {ind}", img)

        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == 32:
            dt = str(datetime.datetime.now())  # 获取当前时间
            filename = f"data/{dt}.pkl"  # 生成文件名
            with open(filename, "wb") as f:
                pickle.dump(results, f)  # 保存结果到文件
                print(f"Successfully save file {filename}")  # 打印保存成功信息
    # print(color_img.shape)
    # print(pcd.shape)
    # yolo_img, yolo_results = yyd.detect(source=color_img,
    #                                     weights="best.pt")
    # print("test")
    for pipeline in rs_pipelines:
        pipeline.stop()
