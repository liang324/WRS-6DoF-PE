# 窄相位射线-三角形相交

import numpy as np
from ..constants import tol
from ..util import diagonal_dot


def rays_triangles_id(triangles,
                      rays,
                      ray_candidates=None,
                      return_any=False):
    '''
    计算一组射线与三角形的相交情况

    :param triangles: (n, 3, 3) 的浮点数组,表示三角形的顶点
    :param rays: (m, 2, 3) 的浮点数组,表示射线的起点和方向
    :param ray_candidates: (m, *) 的整数数组,表示每条射线的候选三角形
    :param return_any: 布尔值,如果为 True,则在任意射线与三角形相交时提前退出循环,并将函数输出更改为布尔值
    :return: 如果 return_any 为 True,返回一个布尔值,表示是否存在相交；否则返回一个数组,表示每条射线与哪些三角形相交
    '''

    # 默认查询的候选三角形集是所有三角形.这会很慢
    candidates = np.ones(len(triangles), dtype=bool)
    hits = [None] * len(rays)

    for ray_index, ray in enumerate(rays):
        if not (ray_candidates is None):
            candidates = ray_candidates[ray_index]

        # 查询三角形候选集
        hit = ray_triangles(triangles[candidates], *ray)
        if return_any:
            if hit.any(): return True
        else:
            hits[ray_index] = np.array(candidates)[hit]
    if return_any: return False
    return np.array(hits)


def ray_triangles(triangles,
                  ray_origin,
                  ray_direction):
    '''
    计算单条射线与多个三角形的相交情况

    使用 Moller-Trumbore 相交算法
    '''
    candidates = np.ones(len(triangles), dtype=bool)

    # 边向量和顶点位置,格式为 (n,3)
    vert0 = triangles[:, 0, :]
    vert1 = triangles[:, 1, :]
    vert2 = triangles[:, 2, :]
    edge0 = vert1 - vert0
    edge1 = vert2 - vert0

    # P 是一个垂直于射线方向和一个三角形边缘的向量
    P = np.cross(ray_direction, edge1)
    # 如果行列式接近于零,射线位于三角形的平面内
    det = diagonal_dot(edge0, P)
    candidates[np.abs(det) < tol.zero] = False

    if not candidates.any():
        return candidates
    # 移除不再是候选的先前计算的项
    inv_det = 1.0 / det[candidates]
    T = ray_origin - vert0[candidates]
    u = diagonal_dot(T, P[candidates]) * inv_det
    new_candidates = np.logical_not(np.logical_or(u < -tol.zero, u > (1 + tol.zero)))
    candidates[candidates] = new_candidates
    if not candidates.any():
        return candidates
    inv_det = inv_det[new_candidates]
    T = T[new_candidates]
    u = u[new_candidates]

    Q = np.cross(T, edge0[candidates])
    v = np.dot(ray_direction, Q.T) * inv_det

    new_candidates = np.logical_not(np.logical_or((v < -tol.zero), (u + v > (1 + tol.zero))))

    candidates[candidates] = new_candidates
    if not candidates.any():
        return candidates

    Q = Q[new_candidates]
    inv_det = inv_det[new_candidates]

    t = diagonal_dot(edge1[candidates], Q) * inv_det
    candidates[candidates] = t > tol.zero

    return candidates
