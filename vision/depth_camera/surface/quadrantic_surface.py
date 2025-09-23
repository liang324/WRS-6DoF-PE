import numpy as np
from scipy.linalg import lstsq
import vision.depth_camera.surface._surface as sfc


class QuadraticSurface(sfc.Surface):
    """
    二次曲面拟合类

    author: ruishuang
    date: 20210625
    """
    def __init__(self,
                 xydata,
                 zdata):
        super().__init__(xydata, zdata)
        # 构建用于线性回归的矩阵 A,其中包含常数项、xy 坐标、xy 乘积项和平方项
        A = np.c_[np.ones(xydata.shape[0]), xydata, np.prod(xydata, axis=1), xydata ** 2]
        # 使用最小二乘法拟合二次曲面,得到系数
        self.coef, _, _, _ = lstsq(A, zdata)

    @staticmethod
    def func(xydata, *parameters):
        """
        计算给定参数下的二次曲面 z 值

        :param xydata: 二维坐标数据
        :param parameters: 二次曲面参数 (a, x, y, xy, x2, y2)
        :return: 计算得到的 z 值
        """
        xydata = np.asarray(xydata)

        def quad(xdata, ydata, a, x, y, xy, x2, y2):
            # 二次曲面方程 z = a + x * xdata + y * ydata + xy * xdata * ydata + x2 * xdata^2 + y2 * ydata^2
            return a + x * xdata + y * ydata + xy * xdata * ydata + x2 * xdata ** 2 + y2 * ydata ** 2

        z = np.zeros(len(xydata))
        # 对每组参数计算二次曲面 z 值并累加
        for single_parameters in np.array(parameters).reshape(-1, 6):
            z += quad(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        """
        获取给定 xydata 对应的 z 值

        :param xydata: 二维坐标数据
        :return: 计算得到的 z 值
        """
        zdata = QuadraticSurface.func(xydata, self.coef)
        return zdata
