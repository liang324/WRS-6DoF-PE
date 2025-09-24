# 作者:陈浩(chen960216@gmail.com 20221113)
# 程序手动标定相机


__VERSION__ = '0.0.1'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr

from robot_sim.robots.robot_interface import RobotInterface


def py2json_data_formatter(data):
    """
    将 Python 数据格式化为 JSON 格式
    仅支持 np.ndarray, str, int, float, dict, list 类型的数据

    :param data: 要格式化的数据
    :return: 格式化后的数据
    """
    # 如果是 numpy 数组,将其转换为列表
    if isinstance(data, np.ndarray):
        return data.tolist()
    # 如果是字符串、浮点数、整数或字典,直接返回
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    # 如果是路径对象,将其转换为字符串
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        # 如果是列表,递归地格式化每个元素
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    """
    使用 JSON 格式导出数据

    :param data: 要导出的数据
    :param path: 导出文件的路径
    :param reminder: 是否提醒文件已存在
    :return: 成功返回 True,失败返回 False
    """
    path = str(path)
    if reminder and os.path.exists(path):
        # 如果文件已存在并且需要提醒,询问用户是否覆盖
        option = input(f"File {path} exists.是否确定要覆盖？, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        # 将数据格式化为 JSON 并写入文件
        json.dump(py2json_data_formatter(data), f)
    return True


class ManualCalibrationBase(ABC):
    def __init__(self, rbt_s: RobotInterface, rbt_x, sensor_hdl, init_calib_mat: np.ndarray = None,
                 component_name="arm", move_resolution=.001, rotation_resolution=np.radians(5)):
        """
        手动校准点云数据的基类

        通过调整机器人和传感器之间的校准矩阵来实现手动校准

        :param rbt_s: 模拟机器人接口
        :param rbt_x: 实际机器人接口
        :param sensor_hdl: 传感器处理器
        :param init_calib_mat: 初始校准矩阵.如果为 None,则使用单位矩阵作为初始校准矩阵
        :param component_name: 安装相机的组件名称
        :param move_resolution: 手动移动调整的分辨率
        :param rotation_resolution: 手动旋转调整的分辨率
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = np.eye(4) if init_calib_mat is None else init_calib_mat
        self._component_name = component_name

        # 存储机器人和点云的绘图节点
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        # 初始化按键映射
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # 添加任务
        taskMgr.doMethodLater(.05, self.sync_rbt, "同步机器人", )
        taskMgr.doMethodLater(.02, self.adjust, "手动调整点云")
        taskMgr.doMethodLater(.5, self.sync_pcd, "同步点云", )

    @abstractmethod
    def get_pcd(self) -> np.ndarray:
        """
        获取点云的抽象方法
        :return: 表示点云的 Nx3 ndarray
        """
        pass

    @abstractmethod
    def get_rbt_jnt_val(self) -> np.ndarray:
        """
        获取机器人关节角度的抽象方法
        :return: 1xn ndarray,n 是机器人的自由度
        """
        pass

    @abstractmethod
    def align_pcd(self, pcd) -> np.ndarray:
        """
        根据校准矩阵对点云进行对齐的抽象方法
        实现手眼或眼手转换
        :return: 表示对齐点云的 Nx3 ndarray
        https://support.zivid.com/en/latest/academy/applications/hand-eye/system-configurations.html
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        通过移动修正校准矩阵的抽象方法
        :param dir: 基于校准矩阵坐标的局部移动方向
        :param dir_global: 基于世界坐标的全局移动方向
        :return:
        """
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        通过旋转修正校准矩阵的抽象方法
        :param dir: 校准矩阵的局部方向
        :param dir_global: 全局方向
        :return:
        """
        self._init_calib_mat[:3, :3] = np.dot(rm.rotmat_from_axangle(dir_global, np.radians(self.rotation_resolution)),
                                              self._init_calib_mat[:3, :3])

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n'):
        def add_key(keys: str or list):
            """
            将按键添加到按键映射中.默认按键映射可以在 visualization/panda/inputmanager.py 中查看
            :param keys: 添加到按键映射中的按键
            """
            assert isinstance(keys, str) or isinstance(keys, list)

            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw

    def sync_pcd(self, task):
        """
        同步真实机器人和模拟机器人
        :return: None
        """
        self._pcd = self.get_pcd()
        self.plot()
        return task.again

    def sync_rbt(self, task):
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(self._component_name, rbt_jnt_val)
        self.plot()
        return task.again

    def save(self):
        """
        保存手动校准结果
        :return:
        """
        dump_json({'affine_mat': self._init_calib_mat.tolist()}, "manual_calibration.json", reminder=False)

    def plot(self, task=None):
        """
        绘制点云和机器人的任务
        :param task:
        :return:
        """
        # 清除之前的绘图
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel()
        self._plot_node_rbt.attach_to(base)
        pcd = self._pcd
        if pcd is not None:
            if pcd.shape[1] == 6:
                pcd, pcd_color = pcd[:, :3], pcd[:, 3:6]
                pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = np.array([1, 1, 1, 1])
            pcd_r = self.align_pcd(pcd)
            self._plot_node_pcd = gm.gen_pointcloud(pcd_r, rgbas=pcd_color_rgba)
            gm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
        if task is not None:
            return task.again

    def adjust(self, task):
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x')
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_')
        elif base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y')
        elif base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_')
        elif base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z')
        elif base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_')
        elif base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x_cw')
        elif base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
        elif base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y_cw')
        elif base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_ccw')
        elif base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z_cw')
        elif base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_ccw')
        else:
            return task.again
        self.plot()
        self.save()
        return task.again


class XArmLite6ManualCalib(ManualCalibrationBase):
    """
    手眼系统的示例实现,适用于 XArmLite6 机器人
    """

    def get_pcd(self):
        """
        获取点云数据并返回带颜色信息的点云

        :return: 包含点云和颜色信息的 Nx6 ndarray
        """
        pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth()
        return np.hstack((pcd, pcd_color))

    def get_rbt_jnt_val(self):
        """
        获取机器人的关节角度

        :return: 1xn ndarray,n 是机器人的自由度
        """
        return self._rbt_x.get_jnt_values()

    def align_pcd(self, pcd):
        """
        根据机器人当前姿态和初始校准矩阵对点云进行对齐

        :param pcd: 原始点云数据
        :return: 对齐后的点云数据
        """
        r2cam_mat = self._init_calib_mat  # 初始校准矩阵
        rbt_pose = self._rbt_x.get_pose()  # 获取机器人当前姿态
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)  # 从机器人姿态生成世界到机器人坐标系的变换矩阵
        w2c_mat = w2r_mat.dot(r2cam_mat)  # 计算世界到相机坐标系的变换矩阵
        return rm.homomat_transform_points(w2c_mat, points=pcd)  # 对点云进行变换


if __name__ == "__main__":
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from drivers.devices.realsense.realsense_d400s import RealSenseD405
    import basis.robot_math as rm
    from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
    from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405()
    # the first frame contains no log_filename information
    rs_pipe.get_pcd_texture_depth()
    rs_pipe.get_pcd_texture_depth()
    rbtx = XArmLite6X(ip='192.168.1.190', has_gripper=False)
    rbt = XArmLite6WRSGripper()

    xarm_mc = XArmLite6ManualCalib(rbt_s=rbt, rbt_x=rbtx, sensor_hdl=rs_pipe)
    base.run()
