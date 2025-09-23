from scipy.interpolate import RBFInterpolator
import vision.depth_camera.surface._surface as sfc


class RBFSurface(sfc.Surface):
    """
    基于径向基函数(RBF)的表面拟合类
    """
    def __init__(self,
                 xydata,
                 zdata,
                 neighbors=None,
                 smoothing=0.0,
                 kernel='thin_plate_spline',
                 epsilon=None,
                 degree=None):
        """
        初始化 RBFSurface 对象

        :param xydata: 二维坐标数据
        :param zdata: 对应的 z 值数据
        :param neighbors: 邻居点的数量,用于局部插值
        :param smoothing: 平滑因子,控制插值的平滑程度
        :param kernel: 使用的核函数类型,例如 'thin_plate_spline'(薄板样条)
        :param epsilon: 核函数的形状参数
        :param degree: 多项式的度数
        :param rng: 范围,格式为 [[xmin, xmax], [ymin, ymax]],如果为 None,则使用 xydata 的最小最大值

        author: weiwei
        date: 20210624
        """
        super().__init__(xydata, zdata)
        self._surface = RBFInterpolator(xydata,
                                        zdata,
                                        neighbors=neighbors,
                                        smoothing=smoothing,
                                        kernel=kernel,
                                        epsilon=epsilon,
                                        degree=degree)

    def get_zdata(self, xydata):
        """
        获取给定 xydata 对应的 z 值

        :param xydata: 二维坐标数据
        :return: 计算得到的 z 值
        """
        return self._surface(xydata)
