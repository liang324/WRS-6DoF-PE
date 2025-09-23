import numpy as np
import modeling.geometric_model as gm


class Surface(object):
    """
    表示一个三维表面的类

    :param xydata: 二维数组,包含表面上的 (x, y) 坐标
    :param zdata: 一维数组,包含对应的 z 坐标数据
    """

    def __init__(self, xydata, zdata):
        self.xydata = xydata  # 存储 (x, y) 坐标数据
        self.zdata = zdata  # 存储 z 坐标数据

    def get_zdata(self, domain):
        """
        根据给定的域获取 z 坐标数据

        :param domain: 指定的域
        :return: z 坐标数据
        :raises NotImplementedError: 如果没有实现该方法
        """

        raise NotImplementedError

    def get_gometricmodel(self,
                          rng=None,
                          granularity=.003,
                          rgba=[.7, .7, .3, 1]):
        """
        获取几何模型

        :param rng: 范围,默认为 xydata 的最小和最大值
        :param granularity: 表面生成的粒度
        :param rgba: 表面颜色的 RGBA 值
        :return: 表面的几何模型
        """
        if rng is None:
            rng = [[min(self.xydata[:, 0]), max(self.xydata[:, 0])],
                   [min(self.xydata[:, 1]), max(self.xydata[:, 1])]]
        surface_gm = gm.gen_surface(self.get_zdata, rng=rng, granularity=granularity)
        surface_gm.set_rgba(rgba=rgba)
        return surface_gm
