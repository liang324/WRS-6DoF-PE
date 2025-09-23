import numpy as np
from ..util import three_dimensionalize, unitize
from ..constants import tol_path as tol


def line_line(origins, directions):
    '''
    找到两条直线之间的交点

    :param origins: (2,d) 直线上的点的列表 (d 可以是 [2,3])
    :param directions: (2,d) 方向向量的列表
    :return: 布尔值和交点坐标.布尔值表示直线是否相交,交点坐标为 (d) 长度的交点或 None.
    '''
    # 将输入的点和方向向量扩展为三维
    is_2D, origins = three_dimensionalize(origins)
    is_2D, directions = three_dimensionalize(directions)
    directions = unitize(directions)

    # 检查方向向量是否平行
    if np.sum(np.abs(np.diff(directions, axis=0))) < tol.zero:
        return False, None

    # 使用文档字符串中的符号
    q_0, p_0 = origins
    v, u = directions
    w = p_0 - q_0

    # 由两个方向向量给出的平面的法向量
    plane_normal = unitize(np.cross(u, v))
    # 与两条直线垂直的向量
    v_perp = unitize(np.cross(v, plane_normal))

    # 如果向量从原点到原点在给定的平面上  方向向量,与平面法向量的点积,应该在零的浮点误差范围内
    coplanar = abs(np.dot(plane_normal, w)) < tol.zero
    if not coplanar:
        return False, None

    # 将交点代入直线方程以找到交点
    s_I = (np.dot(-v_perp, w) / np.dot(v_perp, u))

    # 重新代入直线方程以找到该点
    intersection = p_0 + s_I * u
    return True, intersection[:(3 - is_2D)]
