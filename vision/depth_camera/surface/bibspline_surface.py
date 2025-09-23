import scipy.interpolate as si
import vision.depth_camera.surface._surface as sfc
import numpy as np


class BiBSpline(sfc.Surface):

    def __init__(self,
                 xydata,
                 zdata,
                 degree_x=3,
                 degree_y=3):
        """
        表示双三次样条曲面的类,继承自 Surface 类

        :param xydata: 二维数组,包含表面上的 (x, y) 坐标
        :param zdata: 一维数组,包含对应的 z 坐标数据
        :param degree_x: 样条在 x 方向的阶数,范围为 1~5,推荐值为 3
        :param degree_y: 样条在 y 方向的阶数

        author: weiwei
        date: 20210707
        """
        super().__init__(xydata, zdata)

        # 使用 scipy 的 bisplrep 函数计算双三次样条的参数
        self._tck = si.bisplrep(xydata[:, 0],
                                xydata[:, 1],
                                zdata,
                                kx=degree_x,
                                ky=degree_y)

    def get_zdata(self, xydata):
        """
        根据给定的 (x, y) 坐标数据计算 z 坐标

        :param xydata: 二维数组,包含要计算的 (x, y) 坐标
        :return: 一维数组,包含计算得到的 z 坐标
        """
        return_value = []
        for each_xy in xydata:
            # 使用 scipy 的 bisplev 函数计算每个 (x, y) 对应的 z 值
            each_z = si.bisplev(each_xy[0], each_xy[1], self._tck)
            return_value.append(each_z)
        return np.array(return_value)
        # 也可以使用下面的代码直接计算所有 (x, y) 对应的 z 值
        # return si.bisplev(xydata[:, 0], xydata[:, 1], self._tck)
