from shapely.geometry import Polygon, Point, LineString
from rtree import Rtree
from collections import deque
import numpy as np
import networkx as nx
from .. import bounds
from ..geometry import medial_axis as _medial_axis
from ..constants import tol_path as tol
from ..constants import log
from ..points import transform_points
from ..util import transformation_2D, is_sequence
from .traversal import resample_path


def polygons_enclosure_tree(polygons):
    '''
    给定一个 Shapely 多边形的列表,确定哪些是根(即最外层)多边形,哪些代表穿透根曲线的孔
    通过创建一个 R 树进行粗略的碰撞检测,然后进行多边形查询以获得最终结果

    :param polygons: Shapely 多边形列表
    :return: 根节点列表和包含关系的有向图
    '''
    tree = Rtree()
    for i, polygon in enumerate(polygons):
        tree.insert(i, polygon.bounds)
    count = len(polygons)
    g = nx.DiGraph()
    g.add_nodes_from(np.arange(count))
    for i in range(count):
        if polygons[i] is None: continue
        # 首先从 R 树中查询边界框的交集
        for j in tree.intersection(polygons[i].bounds):
            if (i == j): continue
            # 然后进行更精确的多边形包含测试以生成包围树信息
            if polygons[i].contains(polygons[j]):
                g.add_edge(i, j)
            elif polygons[j].contains(polygons[i]):
                g.add_edge(j, i)
    roots = [n for n, deg in list(g.in_degree().items()) if deg == 0]
    return roots, g


def polygons_obb(polygons):
    """
    找到一组 Shapely 多边形的 OBB(定向边界框)

    :param polygons: Shapely 多边形列表
    :return: 变换矩阵数组和矩形数组
    """
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, p in enumerate(polygons):
        transforms[i], rectangles[i] = polygon_obb(p)
    return np.array(transforms), np.array(rectangles)


def polygon_obb(polygon):
    """
    找到 Shapely 多边形的定向边界框(OBB)

    OBB 始终与多边形凸包的一个边对齐
    :param polygon: Shapely 多边形
    :return: 变换矩阵和变换后多边形的范围
    """
    points = np.asanyarray(polygon.exterior.coords)
    return bounds.oriented_bounds_2D(points)


def transform_polygon(polygon, transform, plot=False):
    """
    变换多边形的坐标

    :param polygon: Shapely 多边形或多边形列表
    :param transform: 变换矩阵
    :param plot: 是否绘制变换后的多边形
    :return: 变换后的多边形
    """
    if is_sequence(polygon):
        result = [transform_polygon(p, t) for p, t in zip(polygon, transform)]
    else:
        shell = transform_points(np.array(polygon.exterior.coords), transform)
        holes = [transform_points(np.array(i.coords), transform) for i in polygon.interiors]
        result = Polygon(shell=shell, holes=holes)
    if plot:
        plot_polygon(result)
    return result


def rasterize_polygon(polygon, pitch, angle=0, return_points=False):
    '''
    给定一个 Shapely 多边形,找到相对于定向边界框的给定角度的栅格表示

    :param polygon: Shapely 多边形对象,要栅格化的 Shapely 多边形

    :param pitch: float,栅格的间距,即每个像素的边长
    :param angle: float, optional,在栅格化之前旋转多边形的角度,相对于定向边界框,默认为 0
    :param return_points: bool, optional,如果为 True,则返回栅格化的点,默认为 False
    :return: numpy.ndarray,(n, m) 布尔数组,其中填充区域为 True
    :return: numpy.ndarray,(3, 3) 变换矩阵,用于将多边形移动到由栅格表示覆盖的区域,从 (0, 0) 开始
    '''
    # 计算多边形的定向边界框并应用旋转变换
    rectangle, transform = polygon_obb(polygon)
    transform = np.dot(transform, transformation_2D(theta=angle))
    vertices = transform_polygon(polygon, transform)

    # 旋转后,我们希望将多边形移回第一象限
    transform[0:2, 2] -= np.min(vertices, axis=0)
    vertices -= np.min(vertices, axis=0)

    # 创建多边形并计算其边界
    p = Polygon(vertices)
    bounds = np.reshape(p.bounds, (2, 2))
    offset = bounds[0]
    shape = np.ceil(np.ptp(bounds, axis=0) / pitch).astype(int)
    grid = np.zeros(shape, dtype=bool)

    def fill(ranges):
        # 填充栅格
        ranges = (np.array(ranges) - offset[0]) / pitch
        x_index = np.array([np.floor(ranges[0]),
                            np.ceil(ranges[1])]).astype(int)
        if np.any(x_index < 0): return
        grid[x_index[0]:x_index[1], y_index] = True
        if (y_index > 0): grid[x_index[0]:x_index[1], y_index - 1] = True

    def handler_multi(geometries):
        # 处理多几何体
        for geometry in geometries:
            handlers[geometry.__class__.__name__](geometry)

    def handler_line(line):
        # 处理线段
        fill(line.xy[0])

    def handler_null(data):
        # 空处理函数
        pass

    handlers = {'GeometryCollection': handler_multi,
                'MultiLineString': handler_multi,
                'MultiPoint': handler_multi,
                'LineString': handler_line,
                'Point': handler_null}

    x_extents = bounds[:, 0] + [-pitch, pitch]

    for y_index in range(grid.shape[1]):
        y = offset[1] + y_index * pitch
        test = LineString(np.column_stack((x_extents, [y, y])))
        hits = p.intersection(test)
        handlers[hits.__class__.__name__](hits)

    log.info('栅格多边形到 %s 网格', str(shape))
    return grid, transform


def plot_polygon(polygon, show=True):
    """
    绘制多边形

    :param polygon: 要绘制的多边形对象
    :param show: 是否显示绘图,默认为 True
    """
    import matplotlib.pyplot as plt

    def plot_single(single):
        plt.plot(*single.exterior.xy, color='b')
        for interior in single.interiors:
            plt.plot(*interior.xy, color='r')

    plt.axes().set_aspect('equal', 'datalim')
    if is_sequence(polygon):
        [plot_single(i) for i in polygon]
    else:
        plot_single(polygon)
    if show: plt.show()


def plot_raster(raster, pitch, offset=[0, 0]):
    '''
    绘制栅格表示

    :param raster: (n,m) 布尔数组,表示填充/空区域
    :param pitch: 栅格中每个方块的边长,在笛卡尔空间中
    :param offset: 栅格网格左下角在笛卡尔空间中的偏移量
    '''
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal', 'datalim')
    filled = (np.column_stack(np.nonzero(raster)) * pitch) + offset
    for location in filled:
        plt.gca().add_patch(plt.Rectangle(location,
                                          pitch,
                                          pitch,
                                          facecolor="grey"))


def resample_boundaries(polygon, resolution, clip=None):
    """
    重新采样多边形的边界

    :param polygon: 要重新采样的多边形对象
    :param resolution: 重新采样的分辨率
    :param clip: 用于限制采样点数的范围,默认为 [8, 200]
    :return: 包含外壳和孔的重新采样结果
    """

    def resample_boundary(boundary):
        # 根据分辨率重新采样多边形的外边界或内边界
        count = boundary.length / resolution
        count = int(np.clip(count, *clip))
        return resample_path(boundary.coords, count=count)

    if clip is None:
        clip = [8, 200]
    # 创建一个包含外壳和孔的点序列
    result = {'shell': resample_boundary(polygon.exterior), 'holes': deque()}
    for interior in polygon.interiors:
        result['holes'].append(resample_boundary(interior))
    result['holes'] = np.array(result['holes'])
    return result


def stack_boundaries(boundaries):
    # 如果没有孔,直接返回外壳
    if len(boundaries['holes']) == 0:
        return boundaries['shell']
    # 将外壳和孔堆叠在一起
    result = np.vstack((boundaries['shell'], np.vstack(boundaries['holes'])))
    return result


def medial_axis(polygon, resolution=.01, clip=None):
    '''
    给定一个 Shapely 多边形,基于均匀分布在多边形边界上的点的 Voronoi 图,找到近似的中轴线

    :param polygon: Shapely 多边形对象
    :param resolution: 多边形边界上每个样本之间的目标距离
    :param clip: [最小样本数, 最大样本数],指定非常精细的分辨率可能导致样本数爆炸,
    因此 clip 指定每个边界区域使用的最小和最大样本数.若不限制,可以指定为 [0, np.inf]
    :return: (n,2,2) 线段集合
    '''

    def contains(points):
        # 检查点是否在多边形内
        return np.array([polygon.contains(Point(i)) for i in points])

    boundary = resample_boundaries(polygon=polygon, resolution=resolution, clip=clip)
    # 重新采样多边形的边界
    boundary = stack_boundaries(boundary)
    return _medial_axis(samples=boundary, contains=contains)


class InversePolygon:
    """
    创建一个逆多边形

    主要用途是在给定多边形内的一个点时,找到到多边形边界的最小距离.
    """

    def __init__(self, polygon):
        _DIST_BUFFER = .05
        # 创建一个围绕多边形的盒子
        bounds = (np.array(polygon.bounds))
        bounds += (_DIST_BUFFER * np.array([-1, -1, 1, 1]))
        coord_ext = bounds[np.array([2, 1, 2, 3, 0, 3, 0, 1, 2, 1])].reshape((-1, 2))
        # 将盒子的内部设置为多边形的外部
        coord_int = [np.array(polygon.exterior.coords)]

        # 一个带有外部形状孔的盒子
        exterior = Polygon(shell=coord_ext, holes=coord_int)
        # 将所有内部多边形制作成外部多边形
        interiors = [Polygon(i.coords) for i in polygon.interiors]
        # 将这些多边形保存到一个平面列表中
        self._polygons = np.append(exterior, interiors)

    def distances(self, point):
        '''
        找到从一个点到外部和内部的最小距离

        :param point: (2) 列表或 Shapely 点对象
        :return: (n) 浮点数列表
        '''
        distances = [i.distance(Point(point)) for i in self._polygons]
        return distances

    def distance(self, point):
        '''
        找到从一个点到多边形边界的最小距离

        :param point: (2) 列表或 Shapely 点对象
        :return: float
        '''
        distance = np.min(self.distances(point))
        return distance


def polygon_hash(polygon):
    '''
    Shapely 多边形对象的近似哈希

    :param polygon: Shapely 多边形对象
    :return: (5) 长度的哈希列表,表示输入多边形
    '''
    result = [len(polygon.interiors),
              polygon.convex_hull.area,
              polygon.convex_hull.length,
              polygon.area,
              polygon.length]
    return result


def random_polygon(segments=8, radius=1.0):
    '''
    生成一个具有最大边数和近似半径的随机多边形

    :param segments: int,随机多边形的最大边数
    :param radius: float,所需多边形的近似半径
    :return: Shapely 多边形对象,具有随机外部,没有内部
    '''
    angles = np.sort(np.cumsum(np.random.random(segments) * np.pi * 2) % (np.pi * 2))
    radii = np.random.random(segments) * radius
    points = np.column_stack((np.cos(angles), np.sin(angles))) * radii.reshape((-1, 1))
    points = np.vstack((points, points[0]))
    polygon = Polygon(points).buffer(0.0)
    if is_sequence(polygon):
        return polygon[0]
    return polygon


def polygon_scale(polygon):
    # 计算多边形的缩放比例
    box = np.abs(np.diff(np.reshape(polygon, (2, 2)), axis=0))
    scale = box.max()
    return scale


def path_to_polygon(path, scale=None):
    try:
        polygon = Polygon(path)
    except ValueError:
        return None
    return repair_invalid(polygon, scale)


def repair_invalid(polygon, scale=None):
    """
    给定一个 Shapely 多边形,尝试返回多边形的有效版本.如果找不到,则返回 None

    :param polygon: Shapely 多边形对象
    :param scale: 用于修复的缩放比例
    :return: 修复后的多边形或 None
    """
    # 如果多边形已经有效,立即返回
    if is_sequence(polygon):
        pass
    elif polygon.is_valid:
        return polygon

    # 基本修复涉及向外缓冲多边形 这将修复一部分问题
    basic = polygon.buffer(tol.zero)
    if basic.area < tol.zero:
        return None

    if basic.is_valid:
        log.debug('通过零缓冲恢复无效多边形')
        return basic

    if scale is None:
        scale = polygon_scale(polygon)
    buffered = basic.buffer(tol.buffer * scale)
    unbuffered = buffered.buffer(-tol.buffer * scale)
    if unbuffered.is_valid and not is_sequence(unbuffered):
        log.debug('通过双重缓冲恢复无效多边形')
        return unbuffered
    log.warn('无法恢复多边形！返回 None!')
    return None
