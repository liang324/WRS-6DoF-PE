import numpy as np


def angles_to_threepoint(angles, center, radius, normal=[0, 0, 1]):
    """
    根据给定的角度、中心和半径计算三个点的坐标

    :param angles: 两个角度的列表,表示圆弧的起始和结束角度
    :param center: 圆的中心坐标,形状为 (2,) 的数组
    :param radius: 圆的半径
    :param normal: 圆所在平面的法向量,默认为 [0, 0, 1]
    :return: 三个点的坐标,形状为 (3, 2) 的数组
    """
    if angles[1] < angles[0]:
        angles[1] += np.pi * 2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    points = planar + center
    return points


def is_ccw(points):
    """
    判断给定的一组二维点是否按逆时针方向排列

    :param points: 二维点的坐标,形状为 (n, 2) 的数组
    :return: 布尔值,表示点是否按逆时针方向排列
    """
    xd = np.diff(points[:, 0])
    yd = np.sum(np.column_stack((points[:, 1], points[:, 1])).reshape(-1)[1:-1].reshape((-1, 2)), axis=1)
    area = np.sum(xd * yd) * .5
    ccw = area < 0
    return ccw
