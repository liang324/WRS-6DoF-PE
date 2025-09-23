import numpy as np

from . import util
from .util import diagonal_dot, unitize, is_shape
from .points import point_plane_distance
from .constants import tol


def cross(triangles):
    '''
    返回输入三角形的两条边的叉积

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :return: 两个边向量的叉积,形状为 (n,3)

    '''
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def area(triangles, sum=True):
    '''
    计算输入三角形的面积总和

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :param sum: bool,返回总面积或单个三角形的面积
    :return:
        如果 sum 为 True,返回三角形的总面积,类型为 float
        否则,返回单个三角形的面积,形状为 (n,)
    '''
    crosses = cross(triangles)
    area = (np.sum(crosses ** 2, axis=1) ** .5) * .5
    if sum:
        return np.sum(area)
    return area


def normals(triangles):
    '''
    计算输入三角形的法向量

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :return: 法向量,形状为 (n,3)
    '''
    crosses = cross(triangles)
    normals, valid = unitize(crosses, check_valid=True)
    return normals, valid


def all_coplanar(triangles):
    '''
    给定一个三角形列表,如果它们都共面,则返回 True,否则返回 False

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :return: all_coplanar, bool
    '''
    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(points=triangles[1:].reshape((-1, 3)),
                                     plane_normal=test_normal,
                                     plane_origin=test_vertex)
    all_coplanar = np.all(np.abs(distances) < tol.zero)
    return all_coplanar


def any_coplanar(triangles):
    '''
    给定一个三角形列表,如果第一个三角形与任何后续三角形共面,则返回 True,否则返回 False

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :return: any_coplanar, bool
    '''
    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(points=triangles[1:].reshape((-1, 3)),
                                     plane_normal=test_normal,
                                     plane_origin=test_vertex)
    any_coplanar = np.any(np.all(np.abs(distances.reshape((-1, 3)) < tol.zero), axis=1))
    return any_coplanar


def mass_properties(triangles, density=1.0, skip_inertia=False):
    '''
    计算一组三角形的质量属性

    实现来源:http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

    :param triangles: 三角形的顶点,形状为 (n,3,3)
    :param density: float,密度
    :param skip_inertia: bool,是否跳过惯性计算
    :return: 包含密度、表面积、体积、质量、质心和惯性(如果计算)的字典
    '''
    crosses = cross(triangles)
    surface_area = np.sum(np.sum(crosses ** 2, axis=1) ** .5) * .5
    # 这些是积分的子表达式
    f1 = triangles.sum(axis=1)
    # triangles[:,0,:] 将给出每个三角形的第一个顶点
    # triangles[:,:,0] 将给出每个三角形的 x 坐标
    f2 = (triangles[:, 0, :] ** 2 +
          triangles[:, 1, :] ** 2 +
          triangles[:, 0, :] * triangles[:, 1, :] +
          triangles[:, 2, :] * f1)

    f3 = ((triangles[:, 0, :] ** 3) +
          (triangles[:, 0, :] ** 2) * (triangles[:, 1, :]) +
          (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2) +
          (triangles[:, 1, :] ** 3) +
          (triangles[:, 2, :] * f2))

    g0 = (f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :])
    g1 = (f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :])
    g2 = (f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :])

    integral = np.zeros((10, len(f1)))
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T

    for i in range(3):
        triangle_i = np.mod(i + 1, 3)
        integral[i + 7] = crosses[:, i] * ((triangles[:, 0, triangle_i] * g0[:, i]) +
                                           (triangles[:, 1, triangle_i] * g1[:, i]) +
                                           (triangles[:, 2, triangle_i] * g2[:, i]))

    coefficents = 1.0 / np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])
    integrated = integral.sum(axis=1) * coefficents

    volume = integrated[0]
    center_mass = integrated[1:4] / volume

    result = {'density': density,
              'surface_area': surface_area,
              'volume': volume,
              'mass': density * volume,
              'center_mass': center_mass.tolist()}

    if skip_inertia:
        return result

    inertia = np.zeros((3, 3))
    inertia[0, 0] = integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum())
    inertia[1, 1] = integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum())
    inertia[2, 2] = integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum())
    inertia[0, 1] = (integrated[7] - (volume * np.product(center_mass[[0, 1]])))
    inertia[1, 2] = (integrated[8] - (volume * np.product(center_mass[[1, 2]])))
    inertia[0, 2] = (integrated[9] - (volume * np.product(center_mass[[0, 2]])))
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    inertia[1, 0] = inertia[0, 1]
    inertia *= density

    result['inertia'] = inertia.tolist()

    return result


def windings_aligned(triangles, normals_compare):
    '''
    给定一组三角形和一组法向量,确定它们是否对齐

    :param triangles: (n,3,3) 顶点位置集合,表示三角形的顶点坐标
    :param normals_compare: (n,3) 法向量集合,用于比较三角形的法向量是否对齐
    :return: (n) 布尔列表,指示法向量是否与三角形对齐
    '''
    calculated, valid = normals(triangles)
    difference = diagonal_dot(calculated, normals_compare[valid])
    result = np.zeros(len(triangles), dtype=bool)
    result[valid] = difference > 0.0
    return result


def bounds_tree(triangles):
    '''
    给定一组三角形,为宽相位碰撞检测创建一个 r-tree

    :param triangles: (n, 3, 3) 顶点列表,表示三角形的顶点
    :return: Rtree 对象,用于空间索引
    '''
    from rtree import index
    # 获得3D r-tree索引所需的属性对象
    properties = index.Property()
    properties.dimension = 3
    # 每个三角形的(n,6)交错边框
    tri_bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))
    # 流加载没有获得正确的索引
    tree = index.Index(properties=properties)
    for i, bounds in enumerate(tri_bounds):
        tree.insert(i, bounds)
    return tree


def nondegenerate(triangles):
    '''
    找到所有面积非零的三角形
    退化三角形分为两种情况:
    1) 三个顶点中有两个重合
    2) 三个顶点都是唯一的但共线

    :param triangles: (n, 3, 3) 浮点数,表示三角形集合
    :return: (n,) 布尔数组,指示哪些三角形具有面积
    '''
    a, valid_a = unitize(triangles[:, 1] - triangles[:, 0], check_valid=True)
    b, valid_b = unitize(triangles[:, 2] - triangles[:, 0], check_valid=True)

    diff = np.zeros((len(triangles), 3))
    diff[valid_a] = a
    diff[valid_b] -= b

    ok = (np.abs(diff) > tol.merge).any(axis=1)
    ok[np.logical_not(valid_a)] = False
    ok[np.logical_not(valid_b)] = False

    return ok


def closest_point(triangles, points):
    """
    返回每个三角形表面上最接近给定点的点

    实现方法来自《实时碰撞检测》,并使用与“ClosestPtPointTriangle”相同的变量名以避免混淆

    :param triangles: (n, 3, 3) 浮点数,表示空间中的三角形顶点
    :param points: (n, 3) 浮点数,表示空间中的点
    :return: (n, 3) 浮点数,表示每个三角形上最接近每个点的点
    """
    # 检查输入的三角形和点
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError('三角形形状错误')
    if not util.is_shape(points, (len(triangles), 3)):
        raise ValueError('需要相同数量的三角形和点！')
    # 存储最近点的位置
    result = np.zeros_like(points)
    # 哪些点还需要处理
    remain = np.ones(len(points), dtype=bool)
    # 如果我们将它与a (n, 3)点积
    # 与array.sum(axis=1)等价但更快
    ones = [1.0, 1.0, 1.0]
    # 获取每个三角形的三个点
    # 使用与RTCD相同的表示法以避免混淆
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    # 检查P是否在A之外的顶点区域内
    ab = b - a
    ac = c - a
    ap = points - a
    # 这是一个更快的版本:diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    is_a = np.logical_and(d1 < tol.zero, d2 < tol.zero)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # 检查P是否在B之外的顶点区域内
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # 进行逻辑检查
    is_b = (d3 > -tol.zero) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # 检查P是否在AB的边缘区域内,如果是,返回P在A上的投影
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol.zero) &
             (d1 > -tol.zero) &
             (d3 < tol.zero) & remain)
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # 检查P是否在C之外的顶点区域内
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol.zero) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # 检查P是否在AC的边缘区域内,如果是,返回P在AC上的投影
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol.zero) & (d2 > -tol.zero) & (d6 < tol.zero) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # 检查P是否在BC的边缘区域,如果是,返回P在BC上的投影
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol.zero) &
             ((d4 - d3) > - tol.zero) &
             ((d5 - d6) > -tol.zero) & remain)
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # 剩余的点必须在人脸区域内
    if any(remain):
        # 点在face区域内
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # 通过重心坐标计算Q
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)
    return result
