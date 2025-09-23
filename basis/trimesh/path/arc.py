import numpy as np
from ..util import three_dimensionalize, euclidean, unitize
from ..constants import log
from ..constants import tol_path as tol
from ..constants import res_path as res
from .intersections import line_line

try:
    from scipy.optimize import leastsq
except ImportError:
    log.warning('No scipy.optimize for arc fitting!')


def arc_center(points):
    '''
    给定一个圆弧的三个点,找到圆心、半径、法向量和角度

    这利用了控制点之间线段的垂直平分线的交点是圆弧的圆心这一事实

    :param points: (3,d) 点列表,其中 d 可以是 2 或 3,表示二维或三维空间中的点

    :return:
        - center: (d) 圆弧的圆心点
        - radius: float, 圆弧的半径
        - plane_normal: (3) 平面的法向量
        - angle: float, 圆弧扫过的角度
    '''
    # 将二维情况视为三维,Z 值为零
    is_2D, points = three_dimensionalize(points, return_2D=True)
    # 找到三角形的两个边向量
    edge_direction = np.diff(points, axis=0)
    edge_midpoints = (edge_direction * .5) + points[0:2]

    # 三个点定义一个平面,因此我们找到它的法向量
    plane_normal = unitize(np.cross(*edge_direction[::-1]))
    vector_edge = unitize(edge_direction)
    vector_perpendicular = unitize(np.cross(vector_edge, plane_normal))

    intersects, center = line_line(edge_midpoints, vector_perpendicular)

    if not intersects:
        raise ValueError('线段不相交！')

    radius = euclidean(points[0], center)
    vector = unitize(points - center)
    angle = np.arccos(np.clip(np.dot(*vector[[0, 2]]), -1.0, 1.0))
    large_arc = (abs(angle) > tol.zero and np.dot(*edge_direction) < 0.0)

    if large_arc:
        angle = (np.pi * 2) - angle

    angles = np.arctan2(*vector[:, 0:2].T[::-1]) + np.pi * 2
    angles_sorted = np.sort(angles[[0, 2]])
    reverse = angles_sorted[0] < angles[1] < angles_sorted[1]
    angles_sorted = angles_sorted[::(1 - int(not reverse) * 2)]

    result = {'center': center[:(3 - is_2D)],
              'radius': radius,
              'normal': plane_normal,
              'span': angle,
              'angles': angles_sorted}
    return result


def discretize_arc(points, close=False, scale=1.0):
    '''
    返回由线段组成的三点圆弧的版本

    :param points: (n, d) 圆弧上的点,其中 d 可以是 2 或 3
    :param close: boolean, 如果为 True,则闭合圆弧(形成圆)

    :return:
        - discrete: (m, d) 离散化的点列表
        - points: (3,3) 或 (3,2) 的点列表,表示从 points[0] 到 points[2] 的圆弧,经过控制点 points[1]
    '''
    two_dimensional, points = three_dimensionalize(points, return_2D=True)
    center_info = arc_center(points)
    center, R, N, angle = (center_info['center'],
                           center_info['radius'],
                           center_info['normal'],
                           center_info['span'])
    if close: angle = np.pi * 2

    # 基于角度标准的面数
    count_a = angle / res.seg_angle
    count_l = ((R * angle)) / (res.seg_frac * scale)

    count = np.max([count_a, count_l])
    # 强制至少 4 个点用于圆弧,否则端点会偏离
    count = np.clip(count, 4, np.inf)
    count = int(np.ceil(count))

    V1 = unitize(points[0] - center)
    V2 = unitize(np.cross(-N, V1))
    t = np.linspace(0, angle, count)

    discrete = np.tile(center, (count, 1))
    discrete += R * np.cos(t).reshape((-1, 1)) * np.tile(V1, (count, 1))
    discrete += R * np.sin(t).reshape((-1, 1)) * np.tile(V2, (count, 1))

    if not close:
        arc_dist = np.linalg.norm(points[[0, -1]] - discrete[[0, -1]], axis=1)
        arc_ok = (arc_dist < tol.merge).all()
        if not arc_ok:
            log.warn('圆弧离散化失败(端点距离 %s)', str(arc_dist))
            log.warn('失败的圆弧点: %s', str(points))
            raise ValueError('圆弧端点偏离!')
    discrete = discrete[:, 0:(3 - two_dimensional)]

    return discrete


def arc_tangents(points):
    '''
    返回点的切向量

    :param points: (n, d) 点列表,其中 d 可以是 2 或 3,表示二维或三维空间中的点
    :return: (n, d) 切向量列表
    '''
    # 将二维情况视为三维,Z 值为零
    two_dimensional, points = three_dimensionalize(points, return_2D=True)
    # 获取圆弧的中心、半径、法向量和角度
    center, R, N, angle = arc_center(points)
    # 计算从中心到点的向量
    vectors = points - center
    # 计算切向量
    tangents = unitize(np.cross(vectors, N))
    return tangents[:, 0:(3 - two_dimensional)]


def arc_offset(points, distance):
    '''
    计算圆弧上点的偏移

    :param points: (n, d) 点列表,其中 d 可以是 2 或 3
    :param distance: float, 偏移距离
    :return: (n, d) 偏移后的点列表
    '''
    # 将二维情况视为三维,Z 值为零
    two_dimensional, points = three_dimensionalize(points)
    # 获取圆弧的中心、半径、法向量和角度
    center, R, N, angle = arc_center(points)
    # 计算从中心到点的单位向量
    vectors = unitize(points - center)
    # 计算偏移后的新点
    new_points = center + vectors * distance
    return new_points[:, 0:(3 - two_dimensional)]


def angles_to_threepoint(angles, center, radius):
    '''
    将角度转换为三点圆弧

    :param angles: (2) 起始和结束角度
    :param center: (d) 圆心
    :param radius: float, 半径
    :return: (3, d) 三点圆弧的点列表
    '''
    if angles[1] < angles[0]: angles[1] += np.pi * 2
    # 计算中间角度
    angles = [angles[0], np.mean(angles), angles[1]]
    # 计算平面上的点
    planar = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    return planar + center


def fit_circle(points, prior=None):
    '''
    使用最小二乘法拟合一组点到一个圆(或 n 维球体)

    :param points: (n, d) 点集
    :param prior:  tuple, 对 (中心, 半径) 的最佳猜测

    :return:
        - center: (d), 圆心的位置
        - radius: float, 圆的平均半径
        - error:  float, 从平均半径的偏差的峰值到峰值的值
    '''

    def residuals(center):
        # 计算每个点到中心的平方距离
        radii_sq = ((points - center) ** 2).sum(axis=1)
        # 计算残差
        residuals = radii_sq - radii_sq.mean()
        return residuals

    if prior is None:
        # 如果没有提供先验,使用点的平均值作为中心的初始猜测
        center_guess = np.mean(points, axis=0)
    else:
        center_guess = prior[0]

    # 使用最小二乘法优化中心位置
    center_result, return_code = leastsq(residuals, center_guess)
    if not (return_code in [1, 2, 3, 4]):
        raise ValueError('最小二乘拟合失败!')

    # 计算每个点到拟合中心的距离
    radii = np.linalg.norm(points - center_result, axis=1)
    # 计算平均半径
    radius = radii.mean()
    # 计算误差
    error = radii.ptp()
    return center_result, radius, error
