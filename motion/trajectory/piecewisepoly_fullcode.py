import scipy.interpolate as sinter
import numpy as np
import math


class PiecewisePoly(object):
    """
    分段多项式类,用于生成线性、三次或五次多项式插值
    Piecewise:“分段的”或“逐段的”.Poly:“Polynomial”的缩写,表示“多项式”
    :param method: 插值方法,支持 "linear"(线性)、"cubic"(三次)和 "quintic"(五次)
    """

    def __init__(self, method="cubic"):
        """
        初始化分段多项式对象
        :param method: 指定插值方法,默认为 "cubic"(三次插值).
                       可选值包括 "linear"(线性插值)和 "quintic"(五次插值).
        """
        self.method = method
        if method == "linear":
            self._solve = self._linear_solve
            self._interpolate = self._linear_interpolate
        elif method == "cubic":
            self._solve = self._cubic_solve
            self._interpolate = self._cubic_interpolate
        elif method == "quintic":
            self._solve = self._quintic_solve
            self._interpolate = self._quintic_interpolate

    def _linear_solve(self):
        """
        线性插值求解器
        """
        x = range(self._path_array.shape[0])
        y = self._path_array
        return sinter.interp1d(x, y, kind="linear", axis=0)

    def _linear_interpolate(self, A, samples):
        """
        线性插值计算
        :param A: 插值函数
        :param samples: 样本点
        :return: 插值结果,包括位置、速度、加速度和插值点
        """
        n_sections = self._n_pnts - 1
        interpolated_x = []
        for i in range(n_sections):
            if i == n_sections - 1:  # 最后一段
                interpolated_x_sec = (samples + i).tolist()
            else:
                interpolated_x_sec = (samples + i).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(np.array(interpolated_x))).tolist()
        interpolated_y_dot = np.zeros_like(interpolated_y).tolist()
        interpolated_y_dotdot = np.zeros_like(interpolated_y).tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    def _cubic_solve(self):
        """ 三次插值求解器 """
        N = self._n_pnts - 1
        X = np.zeros((4 * N, 4 * N))
        Y = np.zeros((4 * N, self._n_dim))
        ridx = 0
        for i in range(N):  # 左端点(共N个)
            X[ridx, i * 4:i * 4 + 4] = np.array([i ** 3, i ** 2, i, 1])
            Y[ridx, :] = self._path_array[i][:]
            ridx += 1
        for i in range(N):  # 右端点(共N个)
            X[ridx, i * 4:i * 4 + 4] = np.array([(i + 1) ** 3, (i + 1) ** 2, i + 1, 1])
            Y[ridx, :] = self._path_array[i + 1][:]
            ridx += 1
        for i in range(N - 1):  # 关键点速度相等
            X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1) ** 2, 2 * (i + 1), 1, 0])
            X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1) ** 2, -2 * (i + 1), -1, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # 关键点加速度相等
            X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1), 1, 0, 0])
            X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1), -1, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        X[-2, :4] = np.array([3 * 0 ** 2, 2 * 0, 1, 0])
        Y[-2, :] = np.zeros(self._n_dim)  # 初始速度为零
        X[-1, -4:] = np.array([3 * (self._n_pnts - 1) ** 2, 2 * (self._n_pnts - 1), 1, 0])
        Y[-1, :] = np.zeros(self._n_dim)  # 结束速度为零
        A = np.linalg.solve(X, Y)
        return A

    def _cubic_interpolate(self, A, samples):
        """
        使用三次多项式进行插值
        :param A: 多项式系数矩阵
        :param samples: 样本点
        :return: 插值后的 y 值、y 的一阶导数、y 的二阶导数,以及插值后的 x 值
        """
        n_sections = self._n_pnts - 1  # 分段数量
        interpolated_x = []  # 插值后的 x 值
        interpolated_y = []  # 插值后的 y 值
        interpolated_y_dot = []  # 插值后的 y 的一阶导数
        interpolated_y_dotdot = []  # 插值后的 y 的二阶导数

        for i in range(n_sections):
            if i == n_sections - 1:  # 最后一段
                interpolated_x_sec = (samples + i).tolist()
            else:
                interpolated_x_sec = (samples + i).tolist()[:-1]
            interpolated_x += interpolated_x_sec

            # 计算 y
            tmp_x_items = np.ones((len(interpolated_x_sec), 4))
            tmp_x_items[:, 0] = np.asarray(interpolated_x_sec) ** 3
            tmp_x_items[:, 1] = np.asarray(interpolated_x_sec) ** 2
            tmp_x_items[:, 2] = np.asarray(interpolated_x_sec)
            interpolated_y_i = tmp_x_items.dot(A[4 * i:4 * i + 4, :])
            interpolated_y += interpolated_y_i.tolist()

            # 计算 y 的一阶导数
            tmp_x_items_dot = np.zeros((len(interpolated_x_sec), 4))
            tmp_x_items_dot[:, 0] = 3 * np.asarray(interpolated_x_sec) ** 2
            tmp_x_items_dot[:, 1] = 2 * np.asarray(interpolated_x_sec)
            tmp_x_items_dot[:, 2] = 1
            interpolated_y_dot_i = tmp_x_items_dot.dot(A[4 * i:4 * i + 4, :])
            interpolated_y_dot += interpolated_y_dot_i.tolist()

            # 计算 y 的二阶导数
            tmp_x_items_dotdot = np.zeros((len(interpolated_x_sec), 4))
            tmp_x_items_dotdot[:, 0] = 6 * np.asarray(interpolated_x_sec)
            tmp_x_items_dotdot[:, 1] = 2
            interpolated_y_dotdot_i = tmp_x_items_dotdot.dot(A[4 * i:4 * i + 4, :])
            interpolated_y_dotdot += interpolated_y_dotdot_i.tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_y, interpolated_x

    def _quintic_solve(self):  # 错误！应该是 6 d
        """
        使用五次多项式求解
        :return: 五次多项式系数矩阵
        """
        N = self._n_pnts - 1  # 分段数量
        X = np.zeros((5 * N, 5 * N))  # 系数矩阵
        Y = np.zeros((5 * N, self._n_dim))  # 结果矩阵
        ridx = 0  # 当前行索引
        for i in range(N):  # 左端点条件 left end (N in total)
            X[ridx, i * 5:i * 5 + 5] = np.array([i ** 4, i ** 3, i ** 2, i, 1])
            Y[ridx, :] = self._path_array[i][:]
            ridx += 1
        for i in range(N):  # 右端点条件 right end (N in total)
            X[ridx, i * 5:i * 5 + 5] = np.array([(i + 1) ** 4, (i + 1) ** 3, (i + 1) ** 2, i + 1, 1])
            Y[ridx, :] = self._path_array[i + 1][:]
            ridx += 1
        for i in range(N - 1):  # # 速度连续性条件 speed 0
            X[ridx, i * 5:i * 5 + 5] = np.array([4 * (i + 1) ** 3, 3 * (i + 1) ** 2, 2 * (i + 1), 1, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-4 * (i + 1) ** 3, -3 * (i + 1) ** 2, -2 * (i + 1), -1, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # # 加速度连续性条件 0
            X[ridx, i * 5:i * 5 + 5] = np.array([6 * (i + 1) ** 2, 3 * (i + 1), 1, 0, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-6 * (i + 1) ** 2, -3 * (i + 1), -1, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # jerk 连续性条件 0
            X[ridx, i * 5:i * 5 + 5] = np.array([4 * (i + 1), 1, 0, 0, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-4 * (i + 1), -1, 0, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        X[-3, :5] = np.array([4 * 0 ** 3, 3 * 0 ** 2, 2 * 0, 1, 0])
        Y[-3, :] = np.zeros(self._n_dim)  # 初始速度为零
        X[-2, -5:] = np.array([4 * (self._n_pnts - 1) ** 3, 3 * (self._n_pnts - 1) ** 2, 2 * (self._n_pnts - 1), 1, 0])
        Y[-2, :] = np.zeros(self._n_dim)  # 终点速度为零
        X[-1, -5:] = np.array([6 * (self._n_pnts - 1) ** 2, 3 * (self._n_pnts - 1), 1, 0, 0])
        Y[-1, :] = np.zeros(self._n_dim)  # 终点加速度为零
        A = np.linalg.solve(X, Y)  # 求解线性方程组
        return A

    def _quintic_interpolate(self, A, samples):
        """
        使用五次多项式进行插值
        :param A: 多项式系数矩阵
        :param samples: 样本点
        :return: 插值后的 y 值、y 的一阶导数、y 的二阶导数,以及插值后的 x 值
        """
        n_sections = self._n_pnts - 1  # 分段数量
        interpolated_x = []  # 插值后的 x 值
        interpolated_y = []  # 插值后的 y 值
        interpolated_y_dot = []  # 插值后的 y 的一阶导数
        interpolated_y_dotdot = []  # 插值后的 y 的二阶导数
        for i in range(n_sections):
            if i == n_sections - 1:  # 最后一段
                interpolated_x_sec = (samples + i).tolist()
            else:
                interpolated_x_sec = (samples + i).tolist()[:-1]
            interpolated_x += interpolated_x_sec
            # 计算 y
            tmp_x_items = np.ones((len(interpolated_x_sec), 5))
            tmp_x_items[:, 0] = np.asarray(interpolated_x_sec) ** 4
            tmp_x_items[:, 1] = np.asarray(interpolated_x_sec) ** 3
            tmp_x_items[:, 2] = np.asarray(interpolated_x_sec) ** 2
            tmp_x_items[:, 3] = np.asarray(interpolated_x_sec)
            interpolated_y_i = tmp_x_items.dot(A[5 * i:5 * i + 5, :])
            interpolated_y += interpolated_y_i.tolist()
            # 计算 y 的一阶导数
            tmp_x_items_dot = np.zeros((len(interpolated_x_sec), 5))
            tmp_x_items_dot[:, 0] = 4 * np.asarray(interpolated_x_sec) ** 3
            tmp_x_items_dot[:, 1] = 3 * np.asarray(interpolated_x_sec) ** 2
            tmp_x_items_dot[:, 2] = 2 * np.asarray(interpolated_x_sec)
            tmp_x_items_dot[:, 3] = 1
            interpolated_y_dot_i = tmp_x_items_dot.dot(A[5 * i:5 * i + 5, :])
            interpolated_y_dot += interpolated_y_dot_i.tolist()
            # 计算 y 的二阶导数
            tmp_x_items_dotdot = np.zeros((len(interpolated_x_sec), 5))
            tmp_x_items_dotdot[:, 0] = 12 * np.asarray(interpolated_x_sec) ** 2
            tmp_x_items_dotdot[:, 1] = 6 * np.asarray(interpolated_x_sec)
            tmp_x_items_dotdot[:, 2] = 2
            interpolated_y_dotdot_i = tmp_x_items_dotdot.dot(A[5 * i:5 * i + 5, :])
            interpolated_y_dotdot += interpolated_y_dotdot_i.tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    def interpolate(self, path, control_frequency=.005, time_interval=1.0, toggle_debug=False):
        """
        对给定路径进行插值
        :param path: 路径点列表
        :param control_frequency: 控制频率,默认值为 0.005
        :param time_interval: 时间间隔,默认值为 1.0
        :param toggle_debug: 是否开启调试模式,默认值为 False
        :return: 插值后的配置、速度、加速度和 x 值
        """
        self._path_array = np.array(path)  # 将路径转换为数组
        self._n_pnts, self._n_dim = self._path_array.shape  # 获取路径点数量和维度
        samples = np.linspace(0,
                              time_interval,
                              math.floor(time_interval / control_frequency),
                              endpoint=True) / time_interval  # 生成样本点
        A = self._solve()  # 求解多项式系数
        interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x = self._interpolate(A, samples)  # 插值
        if toggle_debug:  # 如果开启调试模式
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(3, figsize=(3.5, 4.75))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_confs, 'o')  # 绘制插值后的配置
            for xc in range(self._n_pnts):  # 绘制垂直线以标记路径点
                axs[0].axvline(x=xc)
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_spds)  # 绘制插值后的速度
            axs[2].plot(interpolated_x, interpolated_accs)  # 绘制插值后的加速度
            plt.show()
        return interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x
