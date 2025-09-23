import drivers.rpc.phoxi.phoxi_client as pcdt
import cv2.aruco as aruco
import numpy as np


class SensorMarkerHandler(object):
    """
    传感器标记处理类,用于检测和获取标记的中心位置

    属性:
    - aruco_parameters: ArUco 检测参数
    - aruco_dict: ArUco 字典,用于定义标记类型
    - sensor_client: 传感器客户端,用于与硬件进行通信
    - aruco_target_id_list: 目标 ArUco 标记的 ID 列表

    方法:
    - get_marker_center: 获取标记的中心位置
    """
    def __init__(self):
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.sensor_client = pcdt.PhxClient(host="192.168.125.100:18300")
        self.aruco_target_id_list = [0, 1]

    def get_marker_center(self):
        """
        获取标记的中心位置

        触发传感器获取图像和点云数据,检测 ArUco 标记,并计算标记的中心位置

        :return: 标记在传感器坐标系中的位置,如果检测失败则返回 None
        """
        self.sensor_client.triggerframe()
        img = self.sensor_client.gettextureimg()
        pcd = self.sensor_client.getpcd()
        width = img.shape[1]

        # 检测标记
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_parameters)
        if len(corners) < len(self.aruco_target_id_list) or len(ids) != len(self.aruco_target_id_list):
            return None
        if ids[0] not in self.aruco_target_id_list or ids[1] not in self.aruco_target_id_list:
            return None

        # 计算标记中心
        center = np.mean(np.mean(corners, axis=0), axis=1)[0].astype(np.int32)
        marker_pos_in_sensor = pcd[width * center[1] + center[0]]
        return marker_pos_in_sensor
