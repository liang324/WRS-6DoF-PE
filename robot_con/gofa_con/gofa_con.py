"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
import time
import numpy as np

from .gofa_arm import GoFaArm
from .gofa_state import GoFaState
from .piecewisepoly import PiecewisePoly
from .gofa_constants import GoFaConstants as GFC


class GoFaArmController:
    """
    GoFaArmController 类用于控制 GoFa 机械臂
    """

    def __init__(self, toggle_debug=False, toggle_monitor_only=False):
        """
        初始化 GoFaArmController 类的实例

        :param toggle_debug: 是否启用调试模式,默认为 False.
        :param toggle_monitor_only: 是否仅启用监控模式,默认为 False.
        """
        self._toggle_monitor_only = toggle_monitor_only
        if not toggle_monitor_only:
            self.rbtx = GoFaArm(debug=toggle_debug)
            self._is_add_all = True
            self._traj_opt = PiecewisePoly()
        self.sec_rbtx = GoFaArm(debug=toggle_debug, port=GFC.PORTS['states'])

    @property
    def arm(self):
        """
        返回当前使用的机器人接口.

        :return: 如果 toggle_monitor_only 为 True,则返回次要接口；否则返回主要接口.
        """
        return self.rbtx if not self._toggle_monitor_only else self.sec_rbtx

    def get_pose(self, component_name, return_conf=False):
        raise NotImplementedError

    def get_jnt_values(self):
        """
        获取机械臂关节的角度值

        :return: 1x6 数组,表示关节角度(以弧度为单位).
        author: chen
        """
        return np.deg2rad(self.sec_rbtx.get_state().joints)

    def get_torques(self) -> np.ndarray:
        """
        获取关节的力矩值

        :return: 一个包含关节力矩的数组
        注意: 当关节锁定(空闲)时,力矩为零
        """
        return np.asarray(self.sec_rbtx.get_torques())

    def get_torques_current(self) -> np.ndarray:
        """
        获取当前关节的力矩值
        :return: 一个包含当前关节力矩的数组
        注意: 当关节锁定(空闲)时,力矩为零
        """
        return np.asarray(self.sec_rbtx.get_torques_current())

    def move_j(self, jnt_vals: np.ndarray, speed_n=100, wait=True):
        """
        控制机械臂的关节运动

        :param jnt_vals: 1x7 的 numpy 数组,表示关节角度(以弧度为单位)
        :param speed_n: 速度编号.如果 speed_n = 100,则速度将设置为 RAPID 中指定的 v100,默认为 100.
        :param wait: 是否等待运动完成,默认为 True.
        :return: 布尔值,表示运动是否成功.
        :raises Exception: 如果 toggle_monitor_only 为 True,则抛出异常提示关闭监控模式以启用运动功能.

        author: weiwei
        date: 20170411
        """
        if self._toggle_monitor_only:
            raise Exception("Toggle off monitor only to enable robot movements")
        assert len(jnt_vals) == GoFaState.NUM_JOINTS
        if speed_n == -1:
            self.arm.set_speed_max()
        else:
            speed_data = self.rbtx.get_v(speed_n)
            self.arm.set_speed(speed_data)

        armjnts = np.rad2deg(jnt_vals)
        ajstate = GoFaState(armjnts)
        self.arm.movetstate_sgl(ajstate, wait_for_res=wait)

    def fk(self, component_name: str, jnt_vals: np.ndarray, return_conf: bool = False) -> tuple:
        raise NotImplementedError

    def ik(self, component_name: str,
           pos: np.ndarray,
           rot: np.ndarray,
           conf: np.ndarray = None,
           ext_axis: float = None) -> np.ndarray or None:
        raise NotImplementedError

    def move_jntspace_path(self, path, speed_n=100, wait=True) -> bool:
        """
        控制机械臂沿给定路径在关节空间中运动

        :param path: 关节空间路径.
        :param speed_n: 速度编号.如果 speed_n = 100,则速度将设置为 RAPID 中指定的 v100,默认为 100.
        :param wait: 是否等待运动完成,默认为 True.
        :return: 布尔值,表示运动是否成功.

        :raises Exception: 如果 toggle_monitor_only 为 True,则抛出异常提示关闭监控模式以启用运动功能.
        """
        if self._toggle_monitor_only:
            raise Exception("Toggle off monitor only to enable robot movements")
        statelist = []
        st = time.time()
        # 插值路径以生成状态列表
        for armjnts in self._traj_opt.interpolate_path(path, num=min(100, int(len(path)))):
            armjnts = np.rad2deg(armjnts)
            ajstate = GoFaState(armjnts)
            statelist.append(ajstate)
        et = time.time()
        print("time calculating sending information", et - st)
        # 设置机器人的速度
        if speed_n == -1:
            self.arm.set_speed_max()
        else:
            speed_data = self.arm.get_v(speed_n)
            self.arm.set_speed(speed_data)
        exec_result = self.arm.movetstate_cont(statelist, is_add_all=self._is_add_all, wait_for_res=wait)
        return exec_result

    def stop(self):
        # 停止机械臂的运动
        if not self._toggle_monitor_only:
            self.rbtx.stop()
        self.sec_rbtx.stop()


if __name__ == "__main__":
    yumi_con = GoFaArmController()
    print(yumi_con.get_jnt_values())
