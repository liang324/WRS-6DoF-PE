import numpy as np
import time
from .util import unitize, transformation_2D
from .constants import log
from .grouping import group_vectors
from .points import transform_points, project_to_plane
from .geometry import rotation_2D_to_3D

try:
    from scipy.spatial import ConvexHull
except ImportError:
    log.warning('Scipy import failed!')


def oriented_bounds_2D(points):
    '''
    找到一组二维点的定向边界框

    :param points: (n,2) float,二维点的数组
    :return transform: (3,3) float,齐次二维变换矩阵,用于将输入点集移动到第一象限,因此没有值为负.
    :return rectangle: (2,) float,输入点通过变换后得到的范围大小
    '''
    c = ConvexHull(np.asanyarray(points))
    # (n,2,3) 线段
    hull = c.points[c.simplices]
    # (3,n) 凸包上的点用于检查
    dot_test = c.points[c.vertices].reshape((-1, 2)).T
    edge_vectors = unitize(np.diff(hull, axis=1).reshape((-1, 2)))
    perp_vectors = np.fliplr(edge_vectors) * [-1.0, 1.0]
    bounds = np.zeros((len(edge_vectors), 4))
    for i, edge, perp in zip(range(len(edge_vectors)),
                             edge_vectors,
                             perp_vectors):
        x = np.dot(edge, dot_test)
        y = np.dot(perp, dot_test)
        bounds[i] = [x.min(), y.min(), x.max(), y.max()]

    extents = np.diff(bounds.reshape((-1, 2, 2)), axis=1).reshape((-1, 2))
    area = np.product(extents, axis=1)
    area_min = area.argmin()

    offset = -bounds[area_min][0:2]
    theta = np.arctan2(*edge_vectors[area_min][::-1])

    transform = transformation_2D(offset, theta)
    rectangle = extents[area_min]

    return transform, rectangle


def oriented_bounds(mesh, angle_tol=1e-6):
    '''
    找到一个 Trimesh 的定向边界框

    :param mesh: Trimesh 对象
    :param angle_tol: float,OBB 可以偏离最小体积解的弧度角.即使值较大,返回的范围也会覆盖网格,但体积可能大于最小值
                      较大的值可能会显著加快速度.可接受的值为 >= 0.0 的浮点数.默认值为小(1e-6)但非零
    :return to_origin: (4,4) float,变换矩阵,将输入网格的边界框中心移动到原点.
    :return extents: (3,) float,网格通过 to_origin 变换后的范围
    '''
    # 这个版本的缓存凸包具有指向任意方向的法线(直接来自qhull),使用它可以避免计算昂贵的校正法线.因为法线方向在这里并不重要
    hull = mesh._convex_hull_raw
    vectors = group_vectors(hull.face_normals,
                            angle=angle_tol,
                            include_negative=True)[0]
    min_volume = np.inf
    tic = time.time()
    for i, normal in enumerate(vectors):
        projected, to_3D = project_to_plane(hull.vertices,
                                            plane_normal=normal,
                                            return_planar=False,
                                            return_transform=True)
        height = projected[:, 2].ptp()
        rotation_2D, box = oriented_bounds_2D(projected[:, 0:2])
        volume = np.product(box) * height
        if volume < min_volume:
            min_volume = volume
            rotation_2D[0:2, 2] = 0.0
            rotation_Z = rotation_2D_to_3D(rotation_2D)
            to_2D = np.linalg.inv(to_3D)
            extents = np.append(box, height)
    to_origin = np.dot(rotation_Z, to_2D)
    transformed = transform_points(hull.vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0) * .5)
    to_origin[0:3, 3] = -box_center

    log.debug('oriented_bounds 检查了 %d 个向量,耗时 %0.4fs',
              len(vectors),
              time.time() - tic)
    return to_origin, extents


from .geometry import plane_transform
from .util import vector_to_spherical, is_shape, grid_linspace
from .transformations import translation_matrix, transform_points, spherical_matrix
from .convex import hull_points
from .nsphere import minimum_nsphere
from time import time as now
from scipy import optimize


def minimum_cylinder(obj, sample_count=6, angle_tol=.001):
    """
    找到包含网格或点列表的近似最小体积圆柱体
    采样半球,然后使用 scipy.optimize 选择圆柱体的最终方向
    关于更好实现方法的讨论可以在这里找到 https://www.staff.uni-mainz.de/schoemer/publications/ALGO00.pdf

    :param obj: trimesh.Trimesh 对象,或 (n, 3) float,表示空间中的网格对象或点
    :param sample_count: int,半球的采样密度.角间距为 180 度 / 该数值
    :return result: dict,包含以下键:
        'radius': float,圆柱体的半径
        'height': float,圆柱体的高度
        'transform': (4,4) float,从原点到中心圆柱体的变换
    """

    def volume_from_angles(spherical, return_data=False):
        """
        接受球面坐标并计算沿该向量的圆柱体积

        :param spherical: (2,) float,θ和φ角度
        :param return_data: bool,是否返回详细数据的标志
        :return: 如果 return_data 为 True,返回变换矩阵、半径和高度；否则返回体积
        """
        to_2D = spherical_matrix(*spherical, axes='rxyz')
        projected = transform_points(hull, matrix=to_2D)
        height = projected[:, 2].ptp()

        try:
            center_2D, radius = minimum_nsphere(projected[:, :2])
        except BaseException:
            # 在退化情况下返回无限体积
            return np.inf

        volume = np.pi * height * (radius ** 2)
        if return_data:
            center_3D = np.append(center_2D, projected[
                                             :, 2].min() + (height * .5))
            transform = np.dot(np.linalg.inv(to_2D),
                               translation_matrix(center_3D))
            return transform, radius, height
        return volume

    # 我们被传递了一个具有径向对称性的网格
    # 使用质心和对称轴快速返回结果
    if hasattr(obj, 'symmetry') and obj.symmetry == 'radial':
        # 找到我们的原点
        if obj.is_watertight:
            # 将原点设置为质心
            origin = obj.center_mass
        else:
            # 凸包应该是闭合的
            origin = obj.convex_hull.center_mass
        # 将对称轴与Z轴对齐并将原点移动到零
        to_2D = plane_transform(
            origin=origin,
            normal=obj.symmetry_axis)
        # 将顶点变换到平面以进行检查
        on_plane = transform_points(
            obj.vertices, to_2D)
        # 圆柱体高度是整体Z跨度
        height = on_plane[:, 2].ptp()
        # 平面上的质心是正确的,但沿对称轴的位置可能是错误的,因此需要滑动
        slide = translation_matrix(
            [0, 0, (height / 2.0) - on_plane[:, 2].max()])
        to_2D = np.dot(slide, to_2D)
        # 半径是最大半径
        radius = (on_plane[:, :2] ** 2).sum(axis=1).max() ** 0.5
        # 保存结果
        result = {'height': height,
                  'radius': radius,
                  'transform': np.linalg.inv(to_2D)}
        return result

    # 获取结果的凸包上的点
    hull = hull_points(obj)
    if not is_shape(hull, (-1, 3)):
        raise ValueError('输入必须可以简化为3D点')

    # 采样半球以便局部爬山算法可以进行
    samples = grid_linspace([[0, 0], [np.pi, np.pi]], sample_count)

    # 如果它是旋转对称的,边界圆柱体几乎肯定沿着一个PCI向量
    if hasattr(obj, 'principal_inertia_vectors'):
        # 如果我们有一个网格,添加主惯性向量
        samples = np.vstack(
            (samples,
             vector_to_spherical(obj.principal_inertia_vectors)))

    tic = [now()]
    # 每个样本的投影体积
    volumes = np.array([volume_from_angles(i) for i in samples])
    # 最佳向量在(2,)球面坐标中
    best = samples[volumes.argmin()]
    tic.append(now())

    # 由于我们已经探索了全局空间,将边界设置为仅围绕具有最低体积的样本
    step = 2 * np.pi / sample_count
    bounds = [(best[0] - step, best[0] + step),
              (best[1] - step, best[1] + step)]
    # 运行局部优化
    r = optimize.minimize(volume_from_angles,
                          best,
                          tol=angle_tol,
                          method='SLSQP',
                          bounds=bounds)

    tic.append(now())
    log.debug('在 %f 时间内执行搜索,在 %f 时间内最小化', *np.diff(tic))

    # 实际上分块关于圆柱体的信息
    transform, radius, height = volume_from_angles(r['x'], return_data=True)
    result = {'transform': transform,
              'radius': radius,
              'height': height}
    return result
