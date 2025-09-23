import scipy.interpolate as sinter
import numpy as np
import math
import time
from scipy.optimize import minimize


# 优化过程中忽略插值
class PiecewisePolyOpt(object):
    """
    分段多项式优化类,用于优化路径的时间间隔
    """

    def __init__(self, method="linear"):
        """
        初始化分段多项式优化对象

        :param method: 插值方法,默认为 "linear"(线性插值)
        """
        self._toggle_debug_fine = False
        self._x = None  # 时间点列表
        self._path_array = None  # 路径数组
        self._n_pnts = None  # 路径点数量
        self._n_jnts = None  # 关节数量
        self._control_frequency = None  # 控制频率
        self._start_vels = 0  # 起始速度
        self._start_accs = 0  # 起始加速度
        self._end_vels = 0  # 结束速度
        self._goal_acc = 0  # 目标加速度
        self._max_vels = None  # 最大速度
        self._max_accs = None  # 最大加速度
        self.change_method(method=method)

    def _optimization_goal(self, time_intervals):
        """
        优化目标函数,最小化时间间隔的总和

        :param time_intervals: 时间间隔列表
        :return: 时间间隔的总和
        """
        return np.sum(time_intervals)

    def _constraint_spdacc(self, time_intervals):
        """
        速度和加速度约束

        :param time_intervals: 时间间隔列表
        :return: 违反约束的加速度差的平方和
        """
        self._x = [0]  # 初始化时间点列表
        tmp_total_time = 0  # 临时总时间
        samples_list = []  # 样本列表
        for i in range(len(time_intervals)):
            tmp_time_interval = time_intervals[i]  # 当前时间间隔
            n_samples = math.ceil(tmp_time_interval / .008)  # 样本数量
            if n_samples <= 1:
                n_samples = 2
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  n_samples,
                                  endpoint=True)  # 生成样本点
            samples_list.append(samples + self._x[-1])  # 添加样本点到列表
            self._x.append(tmp_time_interval + tmp_total_time)  # 更新时间点列表
            tmp_total_time += tmp_time_interval  # 更新总时间
        interpolated_x = []  # 插值后的 x 值
        for i in range(len(samples_list)):
            if i == len(samples_list) - 1:  # 最后一段
                interpolated_x_sec = (samples_list[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        A = self._solve()  # 求解多项式系数

        interpolated_y = A(np.array(interpolated_x)).tolist()  # 插值后的 y 值
        interpolated_y_d1 = A(np.array(interpolated_x), 1).tolist()  # 插值后的 y 的一阶导数
        interpolated_y_d2 = A(np.array(interpolated_x), 2).tolist()  # 插值后的 y 的二阶导数
        interpolated_y_d3 = A(np.array(interpolated_x), 3).tolist()  # 插值后的 y 的三阶导数
        original_x = self._x  # 原始时间点

        # 如果开启详细调试模式
        if self._toggle_debug_fine:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, figsize=(35, 47.5))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_y, 'o')  # 绘制插值后的 y 值
            for xc in original_x:
                axs[0].axvline(x=xc)  # 绘制垂直线以标记时间点
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_y_d1)  # 绘制插值后的 y 的一阶导数
            for xc in original_x:
                axs[1].axvline(x=xc)
            axs[2].plot(interpolated_x, interpolated_y_d2)  # 绘制插值后的 y 的二阶导数
            for xc in original_x:
                axs[2].axvline(x=xc)
            axs[3].plot(interpolated_x, interpolated_y_d3)  # 绘制插值后的 y 的三阶导数
            for xc in original_x:
                axs[3].axvline(x=xc)
            for xc in original_x:
                axs[2].axvline(x=xc)
            plt.show()  # 显示图形
        acc_diff = np.tile(self._max_accs, (len(interpolated_y_d2), 1)) - np.abs(interpolated_y_d2)  # 计算加速度差
        # jks_diff = np.tile(self._max_accs, (len(interpolated_y_d3), 1)) - np.abs(interpolated_y_d3)
        print(np.sum(acc_diff[acc_diff < 0] ** 2))  # 输出违反约束的加速度差的平方和
        return np.sum(acc_diff[acc_diff < 0] ** 2)  # 返回违反约束的加速度差的平方和
        # The following code only uses give points
        # for i in range(self._n_pnts - 1):
        #     tmp_time_interval = time_intervals[i]
        #     self._x.append(tmp_time_interval + tmp_total_time)
        #     tmp_total_time += tmp_time_interval
        # A = self._solve()
        # interpolated_y_d2 = A(np.array(self._x), 2).tolist()
        # acc_diff = np.tile(self._max_accs, (len(interpolated_y_d2),1)) - np.abs(interpolated_y_d2)
        # print(np.sum(acc_diff[acc_diff<0]**2))
        # return np.sum(acc_diff[acc_diff<0]**2)

    def _solve_opt(self, method='SLSQP'):
        """
        求解优化问题

        :param tgt_pos: 目标位置(未使用)
        :param tgt_rotmat: 目标旋转矩阵(未使用)
        :param seed_jnt_values: 初始关节值(未使用)
        :param method: 优化方法,默认为 'SLSQP'
        :return: 优化后的时间间隔和目标函数值

        """
        constraints = []  # 约束列表
        constraints.append({'type': 'eq', 'fun': self._constraint_spdacc})  # 添加速度和加速度约束
        bounds = []  # 边界列表
        for i in range(len(self._seed_time_intervals)):
            bounds.append((self._seed_time_intervals[i], None))  # 添加边界
        sol = minimize(self._optimization_goal,
                       self._seed_time_intervals,
                       method=method,
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 10e6, "disp": True},
                       tol=.01)  # 求解优化问题
        if sol.success:  # 如果优化成功
            return sol.x, sol.fun  # 返回优化后的时间间隔和目标函数值
        else:
            return None, None

    def change_method(self, method="cubic"):
        """
        更改插值方法.

        :param method: 插值方法,默认为 "cubic"(三次插值).
        """
        self.method = method  # 设置插值方法
        if method == "linear":
            self._solve = self._linear_solve  # 线性插值
        if method == "quadratic":
            self._solve = self._quadratic_solve  # 二次插值
        elif method == "cubic":
            self._solve = self._cubic_solve  # 三次插值
        elif method == "quintic":
            self._solve = self._quintic_solve  # 五次插值

    def _linear_solve(self):
        """
        线性插值求解

        :return: 插值函数
        """
        return sinter.make_interp_spline(self._x, self._path_array, k=1, axis=0)  # 创建线性插值样条

    def _quadratic_solve(self):
        """
        二次插值求解

        :return: 插值函数
        """
        bc_type = [(1, self._start_vels), None]  # 边界条件:起始速度
        return sinter.make_interp_spline(self._x, self._path_array, k=2, axis=0, bc_type=bc_type)  # 创建二次插值样条

    def _cubic_solve(self):
        """
        三次插值求解

        :return: 插值函数
        """
        bc_type = [(1, self._start_vels), (1, self._end_vels)]  # 边界条件:起始和结束速度
        return sinter.make_interp_spline(self._x, self._path_array, k=3, axis=0, bc_type=bc_type)  # 创建三次插值样条

    def _quintic_solve(self):
        """
        五次插值求解

        :return: 插值函数
        """
        bc_type = [[(1, self._start_vels), (2, self._start_accs)],  # 边界条件:起始速度和加速度
                   [(1, self._end_vels), (2, self._end_accs)]]  # 边界条件:结束速度和加速度
        return sinter.make_interp_spline(self._x, self._path_array, k=5, axis=0, bc_type=bc_type)  # 创建五次插值样条

    def _trapezoid_solve(self):
        """
        梯形插值求解(未实现)
        """
        pass

    def _interpolate(self, A, samples_list):
        """
        插值计算

        :param A: 插值函数
        :param samples_list: 需要插值的样本列表,每个元素是一个 1xn_jnts 的数组
        :return: 插值后的位置、速度、加速度和 x 轴刻度

        author: weiwei
        date: 20210712
        """
        n_sections = self._n_pnts - 1  # 分段数
        interpolated_x = []  # 插值后的 x 轴刻度
        for i in range(n_sections):
            if i == n_sections - 1:  # 最后一段
                interpolated_x_sec = (samples_list[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(interpolated_x)).tolist()  # 插值后的 y 值
        interpolated_y_d1 = A(np.array(interpolated_x), 1).tolist()  # 插值后的速度
        interpolated_y_d2 = A(np.array(interpolated_x), 2).tolist()  # 插值后的加速度
        interpolated_y_d3 = A(np.array(interpolated_x), 3).tolist()  # 插值后的三阶导数
        original_x = self._x  # 原始 x 轴刻度
        return interpolated_y, interpolated_y_d1, interpolated_y_d2, interpolated_y_d3, interpolated_x, original_x

    def _trapezoid_interpolate(self):
        """
        梯形插值计算(未实现)
        """
        pass

    def _remove_duplicate(self, path):
        """
        移除路径中的重复点

        :param path: 路径列表
        :return: 去重后的路径
        """
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):  # 检查是否与下一个点相同
                new_path.append(pose)
        new_path.append(path[-1])  # 添加最后一个点
        return new_path

    def interpolate(self, control_frequency, time_intervals):
        """
        执行插值

        :param control_frequency: 控制频率
        :param time_intervals: 时间间隔列表
        :return: 插值后的配置、速度、加速度、三阶导数、插值 x 轴刻度、原始 x 轴刻度、样本索引
        """
        self._x = [0]  # 初始化 x 轴刻度
        tmp_total_time = 0  # 总时间
        samples_list = []  # 样本列表
        samples_back_index_x = []  # 样本索引
        for i in range(self._n_pnts - 1):
            tmp_time_interval = time_intervals[i]  # 当前时间间隔
            n_samples = math.floor(tmp_time_interval / control_frequency)  # 样本数
            if n_samples <= 1:
                n_samples = 2  # 至少两个样本
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  n_samples,
                                  endpoint=True)  # 生成样本
            for j in range(n_samples):
                samples_back_index_x.append(i)  # 记录样本索引
            samples_list.append(samples + self._x[-1])  # 更新样本列表
            self._x.append(tmp_time_interval + tmp_total_time)  # 更新 x 轴刻度
            tmp_total_time += tmp_time_interval  # 更新总时间
        A = self._solve()  # 求解插值函数
        interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x = \
            self._interpolate(A, samples_list)  # 执行插值
        return interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x, samples_back_index_x

    def interpolate_by_max_spdacc(self,
                                  path,
                                  control_frequency=.005,
                                  start_vels=None,
                                  end_vels=None,
                                  start_accs=None,
                                  end_accs=None,
                                  max_vels=None,
                                  max_accs=None,
                                  toggle_debug_fine=False,
                                  toggle_debug=True):
        """
        根据最大速度和加速度进行插值

        :param path: 路径点列表
        :param control_frequency: 控制频率
        :param start_vels: 起始速度
        :param end_vels: 结束速度
        :param start_accs: 起始加速度
        :param end_accs: 结束加速度
        :param max_vels: 最大关节速度
        :param max_accs: 最大关节加速度
        :param toggle_debug_fine: 是否开启细粒度调试
        :param toggle_debug: 是否开启调试
        :return: 插值后的配置

        author: weiwei
        date: 20210712, 20211012
        """
        self._toggle_debug_fine = toggle_debug_fine
        path = self._remove_duplicate(path)  # 移除重复的路径点
        self._path_array = np.array(path)
        self._n_pnts, self._n_jnts = self._path_array.shape  # 获取路径点和关节数
        self._control_frequency = control_frequency
        if start_vels is None:
            start_vels = [0] * path[0].shape[0]  # 默认起始速度为0
        if start_accs is None:
            start_accs = [0] * path[0].shape[0]  # 默认起始加速度为0
        if end_vels is None:
            end_vels = [0] * path[0].shape[0]  # 默认结束速度为0
        if end_accs is None:
            end_accs = [0] * path[0].shape[0]  # 默认结束加速度为0
        self._start_vels = np.asarray(start_vels)
        self._start_accs = np.asarray(start_accs)
        self._end_vels = np.asarray(end_vels)
        self._end_accs = np.asarray(end_accs)
        if max_vels is None:
            max_vels = [math.pi * 2 / 3] * path[0].shape[0]  # 默认最大速度
        if max_accs is None:
            max_accs = [math.pi] * path[0].shape[0]  # 默认最大加速度
        self._max_vels = np.asarray(max_vels)
        self._max_accs = np.asarray(max_accs)
        # 初始化时间间隔
        time_intervals = []
        for i in range(self._n_pnts - 1):
            pose_diff = abs(path[i + 1] - path[i])  # 计算相邻路径点的差异
            tmp_time_interval = np.max(pose_diff / max_vels)  # 计算时间间隔
            time_intervals.append(tmp_time_interval)
        time_intervals = np.asarray(time_intervals)
        print("初始总时间 seed total time", np.sum(time_intervals))

        # # time scaling
        # # interpolate
        # interpolated_confs, interpolated_vels, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
        #     self.interpolate(control_frequency=control_frequency, time_intervals=time_intervals,
        #                      toggle_debug=toggle_debug_fine)
        # while True:
        #     samples_back_index_x = np.asarray(samples_back_index_x)
        #     interpolated_accs_abs = np.asarray(np.abs(interpolated_accs))
        #     diff_accs = np.tile(max_accs, (len(interpolated_accs_abs), 1)) - interpolated_accs_abs
        #     selection = np.where(np.min(diff_accs, axis=1) < 0)
        #     if len(selection[0]) > 0:
        #         time_intervals += .001
        #         x_sel = np.unique(samples_back_index_x[selection[0] - 1])
        #         time_intervals[x_sel] += .001
        #     else:
        #         break
        #     interpolated_confs, interpolated_vels, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
        #         self.interpolate(control_frequency=control_frequency, time_intervals=time_intervals,
        #                          toggle_debug=toggle_debug_fine)

        # 优化时间间隔
        self._seed_time_intervals = time_intervals
        time_intervals, _ = self._solve_opt()
        # 插值
        interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x, samples_back_index_x = self.interpolate(
            control_frequency=control_frequency, time_intervals=time_intervals)
        print("最终总时间final total time", original_x[-1])
        if toggle_debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, figsize=(35, 47.5))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_confs, 'o')
            for xc in original_x:
                axs[0].axvline(x=xc)
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_vels)
            for xc in original_x:
                axs[1].axvline(x=xc)
            axs[2].plot(interpolated_x, interpolated_accs)
            for xc in original_x:
                axs[2].axvline(x=xc)
            axs[3].plot(interpolated_x, interpolated_jks)
            for xc in original_x:
                axs[3].axvline(x=xc)
            plt.show()
        return interpolated_confs
