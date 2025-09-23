from __future__ import division, print_function
import math
import numpy

__version__ = '2015.03.19'
__docformat__ = 'restructuredtext en'
__all__ = ()


def identity_matrix():
    """
    返回 4x4 单位矩阵
    """
    return numpy.identity(4)


def translation_matrix(direction):
    """
    返回用于按方向向量进行平移的矩阵
    """
    M = numpy.identity(4)
    M[:3, 3] = direction[:3]
    return M


def translation_from_matrix(matrix):
    """
    从平移矩阵中返回平移向量
    """
    return numpy.array(matrix, copy=False)[:3, 3].copy()


def reflection_matrix(point, normal):
    """
    返回用于在由点和法向量定义的平面上镜像的矩阵
    """
    normal = unit_vector(normal[:3])
    M = numpy.identity(4)
    M[:3, :3] -= 2.0 * numpy.outer(normal, normal)
    M[:3, 3] = (2.0 * numpy.dot(point[:3], normal)) * normal
    return M


def reflection_from_matrix(matrix):
    """
    从反射矩阵中返回镜像平面的点和法向量
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    # normal: 对应于特征值 -1 的单位特征向量
    w, V = numpy.linalg.eig(M[:3, :3])
    i = numpy.where(abs(numpy.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有对应于特征值 -1 的单位特征向量")
    normal = numpy.real(V[:, i[0]]).squeeze()
    # point: 任何对应于特征值 1 的单位特征向量
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有对应于特征值 1 的单位特征向量")
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """
    返回用于绕由点和方向定义的轴旋转的矩阵
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # 绕单位向量的旋转矩阵
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[0.0, -direction[2], direction[1]],
                      [direction[2], 0.0, -direction[0]],
                      [-direction[1], direction[0], 0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # 旋转不绕原点
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """
    从旋转矩阵中返回旋转角度和轴
    """
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # 方向: R33 对应于特征值 1 的单位特征向量
    w, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有对应于特征值 1 的单位特征向量")
    direction = numpy.real(W[:, i[-1]]).squeeze()
    # 点: R33 对应于特征值 1 的单位特征向量
    w, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有对应于特征值 1 的单位特征向量")
    point = numpy.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # 旋转角度取决于方向
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
    """
    返回用于在方向上绕原点按因子缩放的矩阵
    使用因子 -1 进行点对称
    """
    if direction is None:
        # 均匀缩放
        M = numpy.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # 非均匀缩放
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = numpy.identity(4)
        M[:3, :3] -= factor * numpy.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * numpy.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """
    从缩放矩阵中返回缩放因子、原点和方向
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    factor = numpy.trace(M33) - 2.0
    try:
        # 方向: 对应于特征值因子的单位特征向量
        w, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(w) - factor) < 1e-8)[0][0]
        direction = numpy.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # 均匀缩放
        factor = (factor + 2.0) / 3.0
        direction = None
    # 原点: 对应于特征值 1 的任意特征向量
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有对应于特征值 1 的特征向量")
    origin = numpy.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
    """
    返回用于投影到由点和法线定义的平面的矩阵

    使用透视点、投影方向或两者都不使用
    如果 pseudo 为 True,透视投影将保持相对深度
    使得 Perspective = dot(Orthogonal, PseudoPerspective)
    """
    M = numpy.identity(4)
    point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # 透视投影
        perspective = numpy.array(perspective[:3], dtype=numpy.float64, copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = numpy.dot(perspective - point, normal)
        M[:3, :3] -= numpy.outer(perspective, normal)
        if pseudo:
            # 保持相对深度
            M[:3, :3] -= numpy.outer(normal, normal)
            M[:3, 3] = numpy.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = numpy.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = numpy.dot(perspective, normal)
    elif direction is not None:
        # 平行投影
        direction = numpy.array(direction[:3], dtype=numpy.float64, copy=False)
        scale = numpy.dot(direction, normal)
        M[:3, :3] -= numpy.outer(direction, normal) / scale
        M[:3, 3] = direction * (numpy.dot(point, normal) / scale)
    else:
        # 正交投影
        M[:3, :3] -= numpy.outer(normal, normal)
        M[:3, 3] = numpy.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """
    从投影矩阵中返回投影平面和透视点

    返回值与 projection_matrix 函数的参数相同:点、法线、方向、透视和伪透视
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point:与特征值1对应的任何特征向量
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # 方向:特征值0对应的单位特征向量
        w, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError("没有特征向量对应于特征值0")
        direction = numpy.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: M33的单位特征向量.T对应于特征值0
        w, V = numpy.linalg.eig(M33.T)
        i = numpy.where(abs(numpy.real(w)) < 1e-8)[0]
        if len(i):
            # 平行投影
            normal = numpy.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # 正交投影,法向量等于方向向量
            return point, direction, None, None, False
    else:
        # 透视投影
        i = numpy.where(abs(numpy.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError("没有特征向量不对应于特征值0")
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / numpy.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """
    返回用于从视锥体获得标准化设备坐标的矩阵

    视锥体边界沿 x(左、右)、y(下、上)和 z(近、远)轴对齐
    如果坐标在视锥体内,标准化设备坐标的范围是 [-1, 1]
    如果 perspective 为 True,视锥体是一个截断的金字塔
    透视点在原点,方向沿 z 轴,否则是一个正交的标准视图体积(一个盒子)
    通过透视剪辑矩阵变换的齐次坐标需要去齐次化(除以 w 坐标)
    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustum")
    if perspective:
        if near <= _EPS:
            raise ValueError("invalid frustum: near <= 0")
        t = 2.0 * near
        M = [[t / (left - right), 0.0, (right + left) / (right - left), 0.0],
             [0.0, t / (bottom - top), (top + bottom) / (top - bottom), 0.0],
             [0.0, 0.0, (far + near) / (near - far), t * far / (far - near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)],
             [0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)],
             [0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)],
             [0.0, 0.0, 0.0, 1.0]]
    return numpy.array(M)


def shear_matrix(angle, direction, point, normal):
    """
    返回用于沿剪切平面上的方向向量按角度剪切的矩阵

    剪切平面由一个点和法向量定义.方向向量必须与平面的法向量正交
    点 P 通过剪切矩阵变换为 P",使得向量 P-P" 平行于方向向量
    并且其长度由 P-P'-P" 的角度决定,其中 P' 是 P 在剪切平面上的正交投影
    """
    normal = unit_vector(normal[:3])
    direction = unit_vector(direction[:3])
    if abs(numpy.dot(normal, direction)) > 1e-6:
        raise ValueError("方向和法向量不是正交的")
    angle = math.tan(angle)
    M = numpy.identity(4)
    M[:3, :3] += angle * numpy.outer(direction, normal)
    M[:3, 3] = -angle * numpy.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """
    从剪切矩阵返回剪切角度、方向和平面
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    # 正态分布:对应于特征值1的交叉独立特征向量
    w, V = numpy.linalg.eig(M33)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("没有找到两个线性无关的特征向量 %s" % w)
    V = numpy.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = numpy.cross(V[i0], V[i1])
        w = vector_norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    normal /= lenorm
    # direction and angle
    direction = numpy.dot(M33 - numpy.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point:特征值1对应的特征向量
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("没有特征向量对应于特征值1")
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """
    从变换矩阵返回一系列变换

    matrix : array_like非退化的齐次变换矩阵
    返回元组:
        scale : 3 个缩放因子的向量
        shear : x-y、x-z、y-z 轴的剪切因子列表
        angles : 关于静态 x、y、z 轴的欧拉角列表
        translate : 沿 x、y、z 轴的平移向量
        perspective : 矩阵的透视部分
    如果矩阵类型错误或退化,则引发 ValueError
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not numpy.linalg.det(P):
        raise ValueError("matrix is singular")
    scale = numpy.zeros((3,))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]
    if any(abs(M[:3, 3]) > _EPS):
        perspective = numpy.dot(M[:, 3], numpy.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = numpy.array([0.0, 0.0, 0.0, 1.0])
    translate = M[3, :3].copy()
    M[3, :3] = 0.0
    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = numpy.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = numpy.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = numpy.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]
    if numpy.dot(row[0], numpy.cross(row[1], row[2])) < 0:
        numpy.negative(scale, scale)
        numpy.negative(row, row)
    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0
    return scale, shear, angles, translate, perspective


def compose_matrix(scale=None, shear=None, angles=None, translate=None, perspective=None):
    """
    从一系列变换中返回变换矩阵

    这是 `decompose_matrix` 函数的逆操作
    变换序列:
        scale : 3 个缩放因子的向量
        shear : x-y、x-z、y-z 轴的剪切因子列表
        angles : 关于静态 x、y、z 轴的欧拉角列表
        translate : 沿 x、y、z 轴的平移向量
        perspective : 矩阵的透视部分
    """
    M = numpy.identity(4)
    if perspective is not None:
        P = numpy.identity(4)
        P[3, :] = perspective[:4]
        M = numpy.dot(M, P)
    if translate is not None:
        T = numpy.identity(4)
        T[:3, 3] = translate[:3]
        M = numpy.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = numpy.dot(M, R)
    if shear is not None:
        Z = numpy.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = numpy.dot(M, Z)
    if scale is not None:
        S = numpy.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = numpy.dot(M, S)
    M /= M[3, 3]
    return M


def orthogonalization_matrix(lengths, angles):
    """
    返回用于晶体学单元坐标的正交化矩阵

    角度以度为单位
    去正交化矩阵是其逆矩阵
    """
    a, b, c = lengths
    angles = numpy.radians(angles)
    sina, sinb, _ = numpy.sin(angles)
    cosa, cosb, cosg = numpy.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return numpy.array([
        [a * sinb * math.sqrt(1.0 - co * co), 0.0, 0.0, 0.0],
        [-a * sinb * co, b * sina, 0.0, 0.0],
        [a * cosb, b * cosa, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """
    返回用于配准两个点集的仿射变换矩阵

    v0 和 v1 是形状为 (ndims, \*) 的数组,至少包含 ndims 个非齐次坐标,
    其中 ndims 是坐标空间的维度
    如果 shear 为 False,则返回相似变换矩阵
    如果 scale 也为 False,则返回刚性/欧几里得变换矩阵
    默认情况下,使用 Hartley 和 Zissermann 的算法
    如果 usesvd 为 True,则根据 Kabsch 的算法通过最小化加权平方偏差和(RMSD)来计算相似和欧几里得变换矩阵
    否则,如果 ndims 为 3,则使用 Horn 的基于四元数的算法,该算法在使用此 Python 实现时较慢
    返回的矩阵执行旋转、平移和统一缩放(如果指定)
    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("输入数组的形状或类型错误")

    # 将质心移动到原点
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # 仿射变换
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2 * ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # 基于协方差矩阵SVD的刚性变换
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # 基于SVD标准正交基的旋转矩阵
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R不构成右手系统
            R -= numpy.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # 齐次变换矩阵
        M = numpy.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # 基于四元数的刚性变换矩阵
        # 计算对称矩阵N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx + yy + zz, 0.0, 0.0, 0.0],
             [yz - zy, xx - yy - zz, 0.0, 0.0],
             [zx - xz, xy + yx, yy - xx - zz, 0.0],
             [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
        # 四元数:对应于大多数正特征值的特征向量
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # 单位四元数
        # 齐次变换矩阵
        M = homomat_from_quaternion(q)
    if scale and not shear:
        # 仿射变换；尺度是RMS偏离质心的比率
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # 将质心移回
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """
    返回用于将给定的 3D 点集转换为第二个点集的矩阵

    v0 和 v1 是形状为 (3, \*) 或 (4, \*) 的数组,至少包含 3 个点
    参数 scale 和 usesvd 在更通用的 affine_matrix_from_points 函数中进行了说明
    返回的矩阵是相似或欧几里得变换矩阵
    该函数在 transformations.c 中有一个快速的 C 实现
    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False, scale=scale, usesvd=usesvd)


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """
    从欧拉角和轴序列返回齐次旋转矩阵

    ai, aj, ak : 欧拉的滚转、俯仰和偏航角
    axes : 24 个轴序列之一,作为字符串或编码元组
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = numpy.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """
    从旋转矩阵返回指定轴序列的欧拉角

    axes : 24 个轴序列之一,作为字符串或编码元组
    请注意,许多欧拉角三元组可以描述一个矩阵
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """
    从四元数返回指定轴序列的欧拉角
    """
    return euler_from_matrix(homomat_from_quaternion(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """
    从欧拉角和轴序列返回四元数

    ai, aj, ak : 欧拉的滚转、俯仰和偏航角
    axes : 24 个轴序列之一,作为字符串或编码元组
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = numpy.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0
    return q


def quaternion_about_axis(angle, axis):
    """
    返回绕轴旋转的四元数
    """
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q


def homomat_from_quaternion(quaternion):
    """
    从四元数返回齐次旋转矩阵
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """
    从旋转矩阵返回四元数

    如果 isprecise 为 True,则输入矩阵被假定为精确的旋转矩阵,并使用更快的算法
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4,))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # 对称矩阵K
        K = numpy.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                         [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                         [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                         [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # 四元数是K的特征向量,对应最大的特征值
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """
    返回两个四元数的乘积
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1

    return numpy.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=numpy.float64)


def quaternion_conjugate(quaternion):
    """
    返回四元数的共轭
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """
    返回四元数的逆
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q / numpy.dot(q, q)


def quaternion_real(quaternion):
    """
    返回四元数的实部
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """
    返回四元数的虚部
    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    返回两个四元数之间的球面线性插值
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = numpy.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        numpy.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quaternion(rand=None):
    """
    返回均匀随机单位四元数

    rand: 类似数组或 None
        三个独立的随机变量,均匀分布在 0 和 1 之间
    """
    if rand is None:
        rand = numpy.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array([numpy.cos(t2) * r2, numpy.sin(t1) * r1,
                        numpy.cos(t1) * r1, numpy.sin(t2) * r2])


def random_rotation_matrix(rand=None):
    """
    返回均匀随机旋转矩阵

    rand: 类似数组
        三个独立的随机变量,均匀分布在 0 和 1 之间,用于每个返回的四元数
    """
    return homomat_from_quaternion(random_quaternion(rand))


class Arcball(object):
    """
    虚拟轨迹球控制
    """

    def __init__(self, initial=None):
        """
        初始化虚拟轨迹球控制

        initial : 四元数或旋转矩阵
        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = numpy.array([0.0, 0.0, 1.0])
        self._constrain = False
        if initial is None:
            self._qdown = numpy.array([1.0, 0.0, 0.0, 0.0])
        else:
            initial = numpy.array(initial, dtype=numpy.float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4,):
                initial /= vector_norm(initial)
                self._qdown = initial
            else:
                raise ValueError("initial not a quaternion or matrix")
        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """
        放置轨迹球,例如当窗口大小改变时

        center : 序列[2] 轨迹球中心的窗口坐标
        radius : float 轨迹球在窗口坐标中的半径
        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """
        设置约束旋转的轴
        """
        if axes is None:
            self._axes = None
        else:
            self._axes = [unit_vector(axis) for axis in axes]

    @property
    def constrain(self):
        """
        返回约束到轴模式的状态
        """
        return self._constrain

    @constrain.setter
    def constrain(self, value):
        """
        设置约束到轴模式的状态
        """
        self._constrain = bool(value)

    def down(self, point):
        """
        设置初始光标窗口坐标并选择约束轴
        """
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
        self._qdown = self._qpre = self._qnow
        if self._constrain and self._axes is not None:
            self._axis = arcball_nearest_axis(self._vdown, self._axes)
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
        else:
            self._axis = None

    def drag(self, point):
        """
        更新当前光标窗口坐标
        """
        vnow = arcball_map_to_sphere(point, self._center, self._radius)
        if self._axis is not None:
            vnow = arcball_constrain_to_axis(vnow, self._axis)
        self._qpre = self._qnow
        t = numpy.cross(self._vdown, vnow)
        if numpy.dot(t, t) < _EPS:
            self._qnow = self._qdown
        else:
            q = [numpy.dot(self._vdown, vnow), t[0], t[1], t[2]]
            self._qnow = quaternion_multiply(q, self._qdown)

    def next(self, acceleration=0.0):
        """
        继续沿最后拖动方向旋转
        """
        q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)
        self._qpre, self._qnow = self._qnow, q

    def matrix(self):
        """
        返回齐次旋转矩阵
        """
        return homomat_from_quaternion(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """
    从窗口坐标返回单位球坐标
    """
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0 * v0 + v1 * v1
    if n > 1.0:
        # 在球体外的位置
        n = math.sqrt(n)
        return numpy.array([v0 / n, v1 / n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])


def arcball_constrain_to_axis(point, axis):
    """
    返回与轴垂直的球面点
    """
    v = numpy.array(point, dtype=numpy.float64, copy=True)
    a = numpy.array(axis, dtype=numpy.float64, copy=True)
    v -= a * numpy.dot(a, v)
    n = vector_norm(v)
    if n > _EPS:
        if v[2] < 0.0:
            numpy.negative(v, v)
        v /= n
        return v
    if a[2] == 1.0:
        return numpy.array([1.0, 0.0, 0.0])
    return unit_vector([-a[1], a[0], 0.0])


def arcball_nearest_axis(point, axes):
    """
    返回与点最近的弧的轴
    """
    point = numpy.array(point, dtype=numpy.float64, copy=False)
    nearest = None
    mx = -1.0
    for axis in axes:
        t = numpy.dot(arcball_constrain_to_axis(point, axis), point)
        if t > mx:
            nearest = axis
            mx = t
    return nearest


# 用于测试一个数字是否接近于零的 epsilon 值
_EPS = numpy.finfo(float).eps * 4.0

# 欧拉角的轴序列
_NEXT_AXIS = [1, 2, 0, 1]

# 将轴字符串映射到/从内轴、奇偶性、重复、框架的元组
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def vector_norm(data, axis=None, out=None):
    """
    返回沿轴的 ndarray 的长度,即欧几里得范数
    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
    """
    返回沿轴的长度归一化的 ndarray,即欧几里得范数
    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data * data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def random_vector(size):
    """
    返回半开区间 [0.0, 1.0) 中随机双精度数组
    """
    return numpy.random.random(size)


def vector_product(v0, v1, axis=0):
    """
    返回垂直于给定向量的向量
    """
    return numpy.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
    """
    返回向量之间的角度

    如果 directed 为 False,则输入向量被解释为无方向轴,即最大角度为 pi/2
    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)
    dot = numpy.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return numpy.arccos(dot if directed else numpy.fabs(dot))


def inverse_matrix(matrix):
    """
    返回方形变换矩阵的逆矩阵
    """
    return numpy.linalg.inv(matrix)


def concatenate_matrices(*matrices):
    """
    返回一系列变换矩阵的连接
    """
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """
    如果两个矩阵执行相同的变换,则返回 True
    """
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1)


def spherical_matrix(theta, phi, axes='sxyz'):
    """
    给定一个球坐标向量,找到将 [0,0,1] 向量转换到这些坐标的旋转矩阵.

    :param theta: float,旋转角度,以弧度为单位
    :param phi: float,旋转角度,以弧度为单位
    :param axes: str,旋转轴的顺序,默认为 'sxyz'
    :return: (4,4) 旋转矩阵,其中以下将是输入球坐标方向的笛卡尔向量:
                np.dot(matrix, [0,0,1,0])
    """
    result = euler_matrix(0.0, phi, theta, axes=axes)
    return result


def transform_points(points,
                     matrix,
                     translate=True):
    """
    返回通过齐次变换矩阵旋转的点

    :param points: (n, D) float,点,其中 D 是 2 或 3
    :param matrix: (3, 3) 或 (4, 4) float,齐次旋转矩阵
    :param translate: bool,是否应用矩阵中的平移
    :return: (n, d) float,变换后的点
    """
    points = numpy.asanyarray(
        points, dtype=numpy.float64)
    # 没有分数就没有哭泣
    if len(points) == 0:
        return points.copy()
    matrix = numpy.asanyarray(matrix, dtype=numpy.float64)
    if (len(points.shape) != 2 or
            (points.shape[1] + 1 != matrix.shape[1])):
        raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
            matrix.shape,
            points.shape))
    # 检查我们是否传递了单位矩阵
    identity = numpy.abs(matrix - numpy.eye(matrix.shape[0])).max()
    if identity < 1e-8:
        return numpy.ascontiguousarray(points.copy())
    dimension = points.shape[1]
    column = numpy.zeros(len(points)) + int(bool(translate))
    stacked = numpy.column_stack((points, column))
    transformed = numpy.dot(matrix, stacked.T).T[:, :dimension]
    transformed = numpy.ascontiguousarray(transformed)
    return transformed
