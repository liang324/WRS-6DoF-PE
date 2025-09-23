import numpy as np
from ..constants import res_path as res
from ..constants import tol_path as tol


def discretize_bezier(points, count=None, scale=1.0):
    '''
    离散化贝塞尔曲线

    :param points: (o,d) bezier 曲线的点列表.第一个和最后一个点应为曲线的起点和终点
                   对于二维三次贝塞尔曲线,阶数 o=3,维度 d=2
    :param count: int, 要离散化的段数.如果未指定,将根据 RES_LENGTH 进行计算
    :param scale: float, 缩放比例

    :return: discrete: (n,d) 点列表,贝塞尔曲线的折线表示,遵循常量 RES_LENGTH
    '''

    def compute(t):
        # 根据采样 t 计算离散点
        t_d = 1.0 - t
        n = len(points) - 1
        # 二项式系数、i 和每个点
        iterable = zip(binomial(n), np.arange(n + 1), points)
        stacked = [((t ** i) * (t_d ** (n - i))).reshape((-1, 1)) * p * c for c, i, p in iterable]
        discrete = np.sum(stacked, axis=0)
        return discrete

    # 确保我们有一个 numpy 数组
    points = np.array(points)

    if count is None:
        # 计算曲线的一小部分需要多少距离
        # 这样我们就可以知道需要多细地采样 t
        norm = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        count = np.ceil(norm / (res.seg_frac * scale))
        count = int(np.clip(count,
                            res.min_sections * len(points),
                            res.max_sections * len(points)))
    result = compute(np.linspace(0.0, 1.0, count))

    test = np.sum((result[[0, -1]] - points[[0, -1]]) ** 2, axis=1)
    assert (test < tol.merge).all()
    assert len(result) >= 2

    return result


def discretize_bspline(control, knots, count=None, scale=1.0):
    '''
    给定 B-样条的控制点和节点向量,返回曲线的采样版本

    :param control: (o,d) B-样条的控制点列表
    :param knots: (j) 节点列表
    :param count: int, 要离散化的段数.如果未指定,将根据 RES_LENGTH 进行计算
    :param scale: float, 缩放比例

    :return: discrete: (count,d) 点列表,B-样条的折线表示
    '''
    # 使用 scipy/fitpack 评估 B-样条
    from scipy.interpolate import splev
    # (n, d) 控制点,其中 d 是顶点的维度
    control = np.array(control)
    degree = len(knots) - len(control) - 1
    if count is None:
        norm = np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
        count = int(np.clip(norm / (res.seg_frac * scale),
                            res.min_sections * len(control),
                            res.max_sections * len(control)))

    ipl = np.linspace(knots[0], knots[-1], count)
    discrete = splev(ipl, [knots, control.T, degree])
    discrete = np.column_stack(discrete)
    return discrete


def binomial(n):
    '''
    返回给定阶数的所有二项式系数

    对于 n > 5,使用 scipy.special.binom,在此我们硬编码以避免 scipy.special 依赖
    '''
    if n == 1:
        return [1, 1]
    elif n == 2:
        return [1, 2, 1]
    elif n == 3:
        return [1, 3, 3, 1]
    elif n == 4:
        return [1, 4, 6, 4, 1]
    elif n == 5:
        return [1, 5, 10, 10, 5, 1]
    else:
        from scipy.special import binom
        return binom(n, np.arange(n + 1))
