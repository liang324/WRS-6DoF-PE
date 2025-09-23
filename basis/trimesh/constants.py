from time import time as time_function
from logging import getLogger as _getLogger
from logging import NullHandler as _NullHandler


# 网格的数值容差
class NumericalToleranceMesh(object):
    '''
    tol.zero: 将小于此值的浮点数视为零
    tol.merge: 合并顶点时,认为距离小于此值的顶点为同一顶点
               根据 SolidWorks 的文档,我们使用相同的值 (1e-8)
    tol.planar: 点可以离平面的最大距离,仍然被认为在平面上
    tol.facet_rsq: 从面中心到相邻面中心画的弧的最小半径平方
                   用于认为两个面共面.此方法比仅考虑法线角度更稳健,
                   因为它对非常小的面上的数值误差具有容忍性
    '''

    def __init__(self, **kwargs):
        # original
        self.zero = 1e-12
        self.merge = 1e-8
        self.planar = 1e-5
        self.facet_rsq = 1e8
        self.fit = 1e-2
        self.id_len = 6
        self.strict = False
        # 管道合并
        # self.zero      = 1e-5
        # self.merge     = 1e-3
        # self.planar    = 1e-3
        # self.facet_rsq = 1e8
        # self.fit       = 1e-2
        # self.id_len    = 6
        self.__dict__.update(kwargs)


class NumericalResolutionMesh(object):
    '''
    res.mesh: 网格化实体时使用的距离
    '''

    def __init__(self, **kwargs):
        self.mesh = 5e-3
        self.__dict__.update(kwargs)


tol = NumericalToleranceMesh()
res = NumericalResolutionMesh()


# 路径的数值容差
class NumericalTolerancePath(object):
    '''
    tol.zero: 将小于此值的浮点数视为零
    tol.merge: 合并顶点时,认为距离小于此值的顶点为同一顶点
    tol.planar: 点可以离平面的最大距离,仍然被认为在平面上
    tol.seg_frac: 简化线段时,线段可以是绘图比例的百分比,并拟合曲线
    tol.seg_angle: 简化线段为弧时,线段可以跨越的角度
    tol.aspect_frac: 简化线段为闭合弧(圆)时,纵横比可以与 1:1 相差的百分比
    tol.radius_frac: 简化线段为弧时,顶点可以偏离拟合半径的百分比
    tol.radius_min: 简化线段为弧时,乘以文档比例的最小半径
    tol.radius_max: 简化线段为弧时,乘以文档比例的最大半径
    tol.tangent: 简化线段为曲线时,末端部分可以偏离切线的最大角度
    '''

    def __init__(self, **kwargs):
        self.zero = 1e-12
        self.merge = 1e-5
        self.planar = 1e-5
        self.buffer = .05
        self.seg_frac = .125
        self.seg_angle = .8
        self.aspect_frac = .1
        self.radius_frac = 1e-2
        self.radius_min = 1e-4
        self.radius_max = 50
        self.tangent = .017
        self.__dict__.update(kwargs)


class NumericalResolutionPath(object):
    '''
    res.seg_frac: 离散化曲线时,单个段应占绘图比例的百分比
    res.seg_angle: 离散化曲线时,段应跨越的角度
    res.max_sections: 离散化样条时,每个控制点的最大段数
    res.min_sections: 离散化样条时,每个控制点的最小段数
    res.export: 导出浮点顶点时使用的格式字符串
    '''

    def __init__(self, **kwargs):
        self.seg_frac = .05
        self.seg_angle = .08
        self.max_sections = 10
        self.min_sections = 5
        self.export = '.5f'


tol_path = NumericalTolerancePath()
res_path = NumericalResolutionPath()

# 日志记录
log = _getLogger('trimesh')
log.addHandler(_NullHandler())


def _log_time(method):
    def timed(*args, **kwargs):
        tic = time_function()
        result = method(*args, **kwargs)
        log.debug('%s 执行时间为 %.4f 秒.',
                  method.__name__,
                  time_function() - tic)
        return result

    timed.__name__ = method.__name__
    timed.__doc__ = method.__doc__
    return timed


# 异常
class MeshError(Exception):
    pass


class TransformError(Exception):
    pass
