import numpy as np
from scipy.linalg import lstsq
import vision.depth_camera.surface._surface as sfc


class PlaneSurface(sfc.Surface):
    """
    平面拟合类

    author: weiwei
    date: 20210707
    """
    def __init__(self,
                 xydata,
                 zdata):
        super().__init__(xydata, zdata)
        # 构建用于线性回归的矩阵 A,其中包含常数项和 xy 坐标
        A = np.c_[np.ones(xydata.shape[0]), xydata[:, 0], xydata[:, 1]]
        # 使用最小二乘法拟合平面,得到系数
        self.coef, _, _, _ = lstsq(A, zdata)

    @staticmethod
    def func(xydata, *parameters):
        """
        计算给定参数下的平面 z 值

        :param xydata: 二维坐标数据
        :param parameters: 平面参数 (a, x, y)
        :return: 计算得到的 z 值
        """
        xydata = np.asarray(xydata)

        def plane(xdata, ydata, a, x, y):
            # 平面方程 z = a + x * xdata + y * ydata
            return a + x * xdata + y * ydata

        z = np.zeros(len(xydata))
        # 对每组参数计算平面 z 值并累加
        for single_parameters in np.array(parameters).reshape(-1, 3):
            z += plane(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        """
        获取给定 xydata 对应的 z 值

        :param xydata: 二维坐标数据
        :return: 计算得到的 z 值
        """
        zdata = PlaneSurface.func(xydata, self.coef)
        return zdata
