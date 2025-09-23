import copy
import numpy as np
import math
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo


# use pip install git+https://github.com/hungpham2511/toppra to install toppra
# webpage is at: https://hungpham2511.github.io/toppra/installation.html

# 使用 pip install git+https://github.com/hungpham2511/toppra 安装 toppra
# 网页地址为: https://hungpham2511.github.io/toppra/installation.html


class PiecewisePolyTOPPRA(object):
    def __init__(self):
        pass

    def _remove_duplicate(self, path):
        # 移除路径中的重复点
        new_path = []
        for i, pose in enumerate(path):
            # 如果当前点与下一个点不相同 ,则添加到新路径中
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        # 添加最后一个点
        new_path.append(path[-1])
        return new_path

    def interpolate_by_max_spdacc(self,
                                  path,
                                  control_frequency=.005,
                                  max_vels=None,
                                  max_accs=None,
                                  toggle_debug=True):
        """
        TODO: 未考虑棱柱电机速度
        :param path: 输入路径
        :param control_frequency: 控制频率 ,默认为0.005
        :param max_vels: 两个相邻姿态之间的最大关节速度 ,如果为None则默认为math.pi
        :param max_accs: 两个相邻姿态之间的最大关节加速度 ,如果为None则默认为math.pi
        :return: 插值后的配置

        TODO: prismatic motor speed is not considered
        :param path:
        :param control_frequency:
        :param max_vels: max jnt speed between two adjacent poses in the path, math.pi if None
        :param max_accs: max jnt speed between two adjacent poses in the path, math.pi if None
        :return:
        author: weiwei
        date: 20210712, 20211012
        """
        # 移除路径中的重复点
        path = self._remove_duplicate(path)
        self._path_array = np.array(path)
        self._n_pnts, _ = self._path_array.shape

        # 如果最大速度或加速度未指定 ,则使用默认值
        if max_vels is None:
            max_vels = [math.pi * 2 / 3] * path[0].shape[0]
        if max_accs is None:
            max_accs = [math.pi] * path[0].shape[0]
        max_vels = np.asarray(max_vels)
        max_accs = np.asarray(max_accs)

        # 初始化种子时间间隔
        # initialize seed time inervals
        time_intervals = []
        for i in range(self._n_pnts - 1):
            pose_diff = abs(path[i + 1] - path[i])
            tmp_time_interval = np.max(pose_diff / max_vels)
            time_intervals.append(tmp_time_interval)
        time_intervals = np.array(time_intervals)
        print("seed total time", np.sum(time_intervals))
        x = [0]
        tmp_total_x = 0
        for i in range(len(time_intervals)):
            tmp_time_interval = time_intervals[i]
            x.append(tmp_time_interval + tmp_total_x)
            tmp_total_x += tmp_time_interval
        interpolated_path = ta.SplineInterpolator(x, path)
        pc_vel = constraint.JointVelocityConstraint(max_vels)
        pc_acc = constraint.JointAccelerationConstraint(max_accs)
        instance = algo.TOPPRA([pc_vel, pc_acc], interpolated_path)
        jnt_traj = instance.compute_trajectory()
        duration = jnt_traj.duration
        print("Found optimal trajectory with duration {:f} sec".format(duration))
        ts = np.linspace(0, duration, math.ceil(duration / control_frequency))
        interpolated_confs = jnt_traj.eval(ts)
        interpolated_spds = jnt_traj.evald(ts)
        interpolated_accs = jnt_traj.evaldd(ts)
        if toggle_debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(3, figsize=(10, 30))
            fig.tight_layout(pad=.7)
            # curve
            axs[0].plot(ts, interpolated_confs, 'o')
            # speed
            axs[1].plot(ts, interpolated_spds)
            for ys in max_vels:
                axs[1].axhline(y=ys)
                axs[1].axhline(y=-ys)
            # acceleration
            axs[2].plot(ts, interpolated_accs)
            for ys in max_accs:
                axs[2].axhline(y=ys)
                axs[2].axhline(y=-ys)
            plt.show()
        return interpolated_confs
