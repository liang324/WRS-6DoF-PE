import numpy as np

from .constants import log, tol
from .grouping import unique_value_in_row
from .util import unitize


def mesh_plane(mesh,
               plane_normal,
               plane_origin=None):
    """
    找到网格与平面之间的交点,返回该平面上的一组线段

    :param mesh: Trimesh对象,表示网格
    :param plane_normal: 1x3 numpy数组,表示平面的法向量
    :param plane_origin: 1x3 numpy数组,表示平面上的一个点
    :return: mx2x3 numpy数组,表示平面上的线段

    author: revised by weiwei
    date: 20210703
    """

    def triangle_cases(signs):
        """
        找出哪个面对应于哪个交集

        每个顶点点积的符号
        通过将每行符号bitbang转换为8位整数来实现这一点

        code : signs      : intersects
        0    : [-1 -1 -1] : No
        2    : [-1 -1  0] : No
        4    : [-1 -1  1] : Yes; 2 on one side, 1 on the other
        6    : [-1  0  0] : Yes; one edge fully on plane
        8    : [-1  0  1] : Yes; one vertex on plane, 2 on different sides
        12   : [-1  1  1] : Yes; 2 on one side, 1 on the other
        14   : [0 0 0]    : No (on plane fully)
        16   : [0 0 1]    : Yes; one edge fully on plane
        20   : [0 1 1]    : No
        28   : [1 1 1]    : No

        :param signs: (n, 3) int数组,值为-1, 0或1
                      每行包含一个面中所有三个顶点与平面的点积符号
        :return:
            basic: (n,) bool数组,基本交点情况的面
            one_vertex: (n,) bool数组,一个顶点情况的面
            one_edge: (n,) bool数组,一个边情况的面
        """
        signs_sorted = np.sort(signs, axis=1)
        coded = np.zeros(len(signs_sorted), dtype=np.int8) + 14
        for i in range(3):
            coded += signs_sorted[:, i] << 3 - i
        # 一条边完全在平面上
        # 注意,我们只接受* *一种* *边缘情况,
        # 另一个顶点的点积为正(16
        # 两种处于边缘的情况([6,16])
        # 这是为了使区域与截面平面共面
        # 我们不会得到无效的边界
        key = np.zeros(29, dtype=bool)
        key[16] = True
        one_edge = key[coded]
        # 一个顶点在平面上,另外两个在不同的边上
        key[:] = False
        key[8] = True
        one_vertex = key[coded]
        # 一个顶点在平面的一边,两个顶点在另一边
        key[:] = False
        key[[4, 12]] = True
        basic = key[coded]
        return basic, one_vertex, one_edge

    def handle_on_vertex(signs, faces, vertices):
        # 处理一个顶点在平面上,两个顶点在不同侧的情况
        vertex_plane = faces[signs == 0]
        edge_thru = faces[signs != 0].reshape((-1, 2))
        point_intersect, valid = plane_lines(plane_origin, plane_normal, vertices[edge_thru.T], line_segments=False)
        lines = np.column_stack((vertices[vertex_plane[valid]], point_intersect)).reshape((-1, 2, 3))
        return lines

    def handle_on_edge(signs, faces, vertices):
        # 处理两个顶点在平面上,一个顶点不在平面的情况
        edges = faces[signs == 0].reshape((-1, 2))
        points = vertices[edges]
        return points

    def handle_basic(signs, faces, vertices):
        # 处理一个顶点在一侧,两个顶点在另一侧的基本情况
        unique_element = unique_value_in_row(signs, unique=[-1, 1])
        edges = np.column_stack((faces[unique_element],
                                 faces[np.roll(unique_element, 1, axis=1)],
                                 faces[unique_element],
                                 faces[np.roll(unique_element, 2, axis=1)])).reshape((-1, 2))
        intersections, valid = plane_lines(plane_origin,
                                           plane_normal,
                                           vertices[edges.T],
                                           line_segments=False)
        # 由于数据已被预先剔除,因此不会有任何无效的交叉点
        # 意味着剔除操作不正确,因此事情变得非常糟糕
        assert valid.all()
        return intersections.reshape((-1, 2, 3))

    # 每个顶点与以face为索引的平面法向量的点积
    # 所以对于每个面,每个顶点的点积是一行
    # 形状与网格相同.面临(n, 3)
    dots = np.dot(plane_normal, (mesh.vertices - plane_origin).T)[mesh.faces]
    # 点积的符号是- 1,0或1
    # 形状与网格相同.面临(n, 3)
    signs = np.zeros(mesh.faces.shape, dtype=np.int8)
    signs[dots < -tol.merge] = -1
    signs[dots > tol.merge] = 1
    # 找出横截面上有哪些三角形
    # 以及它们所在的三种交叉情况中的哪一种
    cases = triangle_cases(signs)
    # 每种情况的处理程序
    handlers = (handle_basic,
                handle_on_vertex,
                handle_on_edge)
    lines = np.vstack([h(signs[c],
                         mesh.faces[c],
                         mesh.vertices) for c, h in zip(cases, handlers)])
    log.debug('mesh_cross_section found %i intersections', len(lines))
    return lines


def plane_lines(plane_origin,
                plane_normal,
                endpoints,
                line_segments=True):
    """
    计算平面与线的交点

    :param plane_origin: 1x3 numpy数组,表示平面上的一个点
    :param plane_normal: 1x3 numpy数组,表示平面的法向量
    :param endpoints: 2xnx3 numpy数组,定义要测试交点的线
    :param line_segments: bool,如果为True,仅返回端点在不同侧的交点
    :return: mx3 numpy数组,表示交点,nx3布尔数组,指示有效交点

    author: revised by weiwei
    date: 20210703
    """
    endpoints = np.asanyarray(endpoints)
    plane_origin = np.asanyarray(plane_origin).reshape(3)
    line_dir = unitize(endpoints[1] - endpoints[0])
    plane_normal = unitize(np.asanyarray(plane_normal).reshape(3))
    t = np.dot(plane_normal, (plane_origin - endpoints[0]).T)
    b = np.dot(plane_normal, line_dir.T)
    # 如果平面法线和直线方向垂直,则意味着向量在平面上,没有有效的交集
    # 我们通过检查点积非零来丢弃平面上的向量
    valid = np.abs(b) > tol.zero
    if line_segments:
        test = np.dot(plane_normal, np.transpose(plane_origin - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        nonzero = np.logical_or(np.abs(t) > tol.zero, np.abs(test) > tol.zero)
        valid = np.logical_and(valid, different_sides)
        valid = np.logical_and(valid, nonzero)
    d = np.divide(t[valid], b[valid])
    intersection = endpoints[0][valid]
    intersection += np.reshape(d, (-1, 1)) * line_dir[valid]
    return intersection, valid
