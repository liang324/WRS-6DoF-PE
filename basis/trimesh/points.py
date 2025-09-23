import numpy as np
from .constants import log, tol
from .geometry import plane_transform


def transform_points(points, matrix, translate=True):
    """
    返回被变换矩阵旋转后的点
    如果点的形状是 (n,2),矩阵必须是 (3,3)
    如果点的形状是 (n,3),矩阵必须是 (4,4)

    :param points: nx2 或 nx3 的点集
    :param matrix: 3x3 或 4x4 的齐次变换矩阵
    :param translate: 是否应用矩阵中的平移
    :return: 变换后的点集

    author: revised by weiwei
    date: 20201202
    """
    points = np.asanyarray(points)
    dimension = points.shape[1]
    column = np.zeros(len(points)) + int(bool(translate))
    stacked = np.column_stack((points, column))
    transformed = np.dot(matrix, stacked.T).T[:, 0:dimension]
    return transformed


def point_plane_distance(points, plane_normal, plane_origin=[0, 0, 0]):
    """
    计算点到平面的距离

    :param points: 点集
    :param plane_normal: 平面的法向量
    :param plane_origin: 平面上的一点,默认为 [0, 0, 0]
    :return: 点到平面的距离
    """
    w = np.array(points) - plane_origin
    distances = np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal)
    return distances


def major_axis(points):
    """
    返回一个近似向量,表示点集的主轴

    :param points: nxd 的点集
    :return: 1xd 的向量

    author: revised by weiwei
    date: 20201202
    """
    u, s, v = np.linalg.svd(points)
    axis_guess = v[np.argmax(s)]
    return axis_guess


def surface_normal(points):
    """
    使用 SVD 返回一组点的法向量估计

    :param points: nxd 的点集
    :return: 1xd 的向量

    author: revised by weiwei
    date: 20201202
    """
    normal = np.linalg.svd(points)[2][-1]
    return normal


def plane_fit_lmse(points, tolerance=None):
    """
    TODO: RANSAC 版本？
    给定一组点,使用最小二乘法找到一个平面上的点和法向量.

    :param points: nx3 的 numpy 数组.
    :param tolerance: 结果的非平面性容忍度.
    :return: C: (3) 平面上的一点, N: (3) 法向量.

    author: revised by weiwei
    date: 20201202
    """
    C = points[0]
    x = points - C
    M = np.dot(x.T, x)
    N = np.linalg.svd(M)[0][:, -1]
    if not (tolerance is None):
        normal_range = np.ptp(np.dot(N, points.T))
        if normal_range > tol.planar:
            log.error('点有 %f 的峰值', normal_range)
            raise ValueError('平面外公差！')
    return C, N


def radial_sort(points, origin=None, normal=None):
    """
    按角度对一组点进行径向排序(围绕一个原点/法向量)
    如果未指定原点/法向量,则围绕质心和点所在的近似平面进行排序

    :param points: nx3 的点集
    :param origin: 排序的原点
    :param normal: 排序的法向量
    :return: 排序后的点集

    author: revised by weiwei
    date: 20201202
    """
    # 如果未指定origin和normal,则在质心处生成一个
    if origin == None: origin = np.average(points, axis=0)
    if normal == None: normal = surface_normal(points)
    # 创建两个相互垂直的坐标轴和法线并将这些点投影到它们上面
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)
    # 计算坐标轴上各点的角度
    angles = np.arctan2(pr0, pr1)
    # 返回按角度排序的点
    return points[[np.argsort(angles)]]


def project_to_plane(points, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0], transform=None, return_transform=False,
                     return_planar=True):
    """
    将一组 nx3 的点投影到一个平面上

    :param points: nx3 的 numpy 数组
    :param plane_normal: 1x3 的 numpy 数组,表示平面的法向量
    :param plane_origin: 1x3 的 numpy 数组,表示平面上的一点
    :param transform: None 或 4x4 的 numpy 数组.如果指定,法向量和原点将被忽略
    :param return_transform: bool,如果为 True,则返回用于将点投影到平面的 4x4 矩阵
    :param return_planar: bool,如果为 True,返回 nx2 的点集.如果为 False,返回 nx3 的点集,其中 Z 列为零
    :return: 投影后的点集,可能还有投影矩阵
    """
    if np.all(np.abs(plane_normal) < tol.zero):
        raise NameError('Normal must be nonzero!')
    if transform is None:
        transform = plane_transform(plane_origin, plane_normal)
    transformed = transform_points(points, transform)
    transformed = transformed[:, 0:(3 - int(return_planar))]
    if return_transform:
        polygon_to_3D = np.linalg.inv(transform)
        return transformed, polygon_to_3D
    return transformed


def absolute_orientation(points_A, points_B, return_error=False):
    """
    计算最佳对齐 points_A 和 points_B 的变换
    使用 Horn 方法解决绝对方向问题,在 3D 中不进行缩放

    :param points_A: nx3 的列表
    :param points_B: nx3 的列表
    :param return_error: boolean,如果为 True,返回一个表示 T*points_A[i] 到 points_B[i] 的欧几里得距离的 1xn 列表
    :return: M: 4x4 的变换矩阵,用于最佳对齐 points_A 到 points_B,error: float,最大欧几里得距离的列表

    author: revised by weiwei
    date: 20201202
    """
    points_A = np.array(points_A)
    points_B = np.array(points_B)
    if (points_A.shape != points_B.shape):
        raise ValueError('Points must be of the same shape!')
    if len(points_A.shape) != 2 or points_A.shape[1] != 3:
        raise ValueError('Points must be (n,3)!')
    lc = np.average(points_A, axis=0)
    rc = np.average(points_B, axis=0)
    left = points_A - lc
    right = points_B - rc
    M = np.dot(left.T, right)
    [[Sxx, Sxy, Sxz],
     [Syx, Syy, Syz],
     [Szx, Szy, Szz]] = M
    N = [[(Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx)],
         [(Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz)],
         [(Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy)],
         [(Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz)]]
    (w, v) = np.linalg.eig(N)
    q = v[:, np.argmax(w)]
    q = q / np.linalg.norm(q)
    M1 = [[q[0], -q[1], -q[2], -q[3]],
          [q[1], q[0], q[3], -q[2]],
          [q[2], -q[3], q[0], q[1]],
          [q[3], q[2], -q[1], q[0]]]
    M2 = [[q[0], -q[1], -q[2], -q[3]],
          [q[1], q[0], -q[3], q[2]],
          [q[2], q[3], q[0], -q[1]],
          [q[3], -q[2], q[1], q[0]]]
    R = np.dot(np.transpose(M1), M2)[1:4, 1:4]
    T = rc - np.dot(R, lc)
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = T
    if return_error:
        errors = np.sum((transform_points(points_A, M) - points_B) ** 2, axis=1)
        return M, errors.max()
    return M


def remove_close_pairs(points, radius):
    """
    给定一个 nxd 的点集,其中 d=2 或 3,返回一个点列表,其中没有点之间的距离小于 radius

    :param points: nxd 的点列表
    :param radius: 距离半径
    :return: 去除近距离点后的点集

    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    # 获取比半径更近的每一对点的索引
    pairs = tree.query_pairs(radius, output_type='ndarray')
    # 每个顶点索引在一对中出现的频率
    # 这本质上是一个低成本计算的“顶点度”
    # 在我们可以为连接点构建的图中
    count = np.bincount(pairs.ravel(), minlength=len(points))
    # 对于每一对,我们必须删除其中一个
    # 我们选择的两个选项中的哪一个会有很大的影响
    # 关于我们最终做了多少过度筛选
    column = count[pairs].argmax(axis=1)
    # 取每一行中次数最高的值
    # 这里可能有更好的numpy切片方法
    highest = pairs.ravel()[column + 2 * np.arange(len(column))]
    # 通过索引屏蔽顶点
    mask = np.ones(len(points), dtype=bool)
    mask[highest] = False
    if tol.strict:
        # 验证我们确实做到了我们说过要做的
        test = cKDTree(points[mask])
        assert len(test.query_pairs(radius)) == 0
    return points[mask], mask


def remove_close_withfaceid(points, face_index, radius):
    """
    给定一个 nxd 的点集,其中 d=2 或 3,返回一个点列表,其中没有点之间的距离小于 radius

    :param points: 点集
    :param face_index: 面索引
    :param radius: 距离半径
    :return: 去除近距离点后的点集及其面索引

    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(points)
    consumed = np.zeros(len(points), dtype=bool)
    unique = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        if consumed[i]:
            continue
        neighbors = tree.query_ball_point(points[i], r=radius)
        consumed[neighbors] = True
        unique[i] = True
    return points[unique], face_index[unique]


def remove_close_between_two_sets(points_fixed, points_reduce, radius):
    """
    给定两个点集和一个半径,返回一个点集,该点集是 points_reduce 的子集,其中没有点在 points_fixed 中任何点的半径范围内

    :param points_fixed: 固定点集
    :param points_reduce: 待减少的点集
    :param radius: 距离半径
    :return: 去除近距离点后的点集

    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree as KDTree
    tree_fixed = KDTree(points_fixed)
    tree_reduce = KDTree(points_reduce)
    reduce_duplicates = tree_fixed.query_ball_tree(tree_reduce, r=radius)
    reduce_duplicates = np.unique(np.hstack(reduce_duplicates).astype(int))
    reduce_mask = np.ones(len(points_reduce), dtype=bool)
    reduce_mask[reduce_duplicates] = False
    points_clean = points_reduce[reduce_mask]
    return points_clean


def k_means(points, k, **kwargs):
    """
    找到 k 个质心,尝试最小化 k-均值问题: https://en.wikipedia.org/wiki/Metric_k-center

    :param points: nxd 的点列表.
    :param k: int,计算的质心数量.
    :param **kwargs: 直接传递给 scipy.cluster.vq.kmeans 的参数.
    :return: centroids: kxd 的质心点列表, labels: 1xn 的索引列表,指示哪些点属于哪个质心.

    author: revised by weiwei
    date: 20201202
    """
    from scipy.cluster.vq import kmeans
    from scipy.spatial import cKDTree
    points = np.asanyarray(points)
    points_std = points.std(axis=0)
    whitened = points / points_std
    centroids_whitened, distortion = kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]
    return centroids, labels


def plot_points(points, show=True):
    """
    绘制点集

    :param points: nxd 的点列表
    :param show: bool,是否显示绘图
    :return: None
    """
    import matplotlib.pyplot as plt
    points = np.asanyarray(points)
    dimension = points.shape[1]
    if dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
    elif dimension == 2:
        plt.scatter(*points.T)
    else:
        raise ValueError('Points must be 2D or 3D, not %dD', dimension)
    if show:
        plt.show()
