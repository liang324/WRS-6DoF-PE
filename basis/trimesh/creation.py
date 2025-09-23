from basis.trimesh.base import Trimesh
from basis.trimesh.constants import log, tol
from basis.trimesh.triangles import normals
from basis.trimesh.geometry import faces_to_edges
from basis.trimesh.grouping import group_rows, unique_rows
from basis.trimesh import util
import numpy as np
from collections import deque

try:
    from shapely.geometry import Polygon
    from shapely.wkb import loads as load_wkb
except ImportError:
    log.warning('shapely unavilable', exc_info=True)


def validate_polygon(obj):
    '''
    验证输入对象是否为有效的多边形

    :param obj: 输入对象,可以是 Polygon 对象、(n, 2) 形状的数组或字符串
    :return polygon: 验证后的 Polygon 对象
    '''
    if util.is_instance_named(obj, 'Polygon'):
        polygon = obj
    elif util.is_shape(obj, (-1, 2)):
        polygon = Polygon(obj)
    elif util.is_string(obj):
        polygon = load_wkb(obj)
    else:
        raise ValueError('输入不是多边形！')

    if (not polygon.is_valid or polygon.area < tol.zero):
        raise ValueError('多边形面积为零或无效！')
    return polygon


def extrude_polygon(polygon,
                    height,
                    **kwargs):
    '''
    将一个二维多边形拉伸为三维网格

    :param polygon: Shapely.geometry.Polygon 对象
    :param height: float,拉伸的高度
    :return mesh: Trimesh 对象,表示拉伸后的三维网格
    '''
    # 创建 Shapely 多边形的二维三角剖分
    vertices, faces = triangulate_polygon(polygon, **kwargs)
    mesh = extrude_triangulation(vertices=vertices,
                                 faces=faces,
                                 height=height,
                                 **kwargs)
    return mesh


def extrude_triangulation(vertices,
                          faces,
                          height,
                          **kwargs):
    '''
    将 Shapely.geometry Polygon 对象和高度(float)转换为密闭的 Trimesh 对象

    :param vertices: (n, 2) float 数组,表示二维顶点
    :param faces: (n, 3) int 数组,表示三角形的顶点索引
    :param height: float,拉伸的高度
    :return mesh: Trimesh 对象,表示拉伸后的三维网格
    '''
    # 确保三角剖分的绕组方向向上
    normal_test = normals([util.three_dimensionalize(vertices[faces[0]])[1]])[0]
    if np.dot(normal_test, [0, 0, 1]) < 0:
        faces = np.fliplr(faces)
    # 将 (n,3) 面堆叠为 (3*n, 2) 边
    edges = faces_to_edges(faces)
    edges_sorted = np.sort(edges, axis=1)
    # 仅出现一次的边位于多边形的边界上
    edges_unique = group_rows(edges_sorted, require_count=1)
    # (n, 2, 2) 线段集合(位置,而不是引用)
    boundary = vertices[edges[edges_unique]]
    # 为二维三角剖分边界上的每个二维线段创建两个垂直三角形
    vertical = np.tile(boundary.reshape((-1, 2)), 2).reshape((-1, 2))
    vertical = np.column_stack((vertical, np.tile([0, height, 0, height], len(boundary))))
    vertical_faces = np.tile([3, 1, 2, 2, 1, 0], (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1, 1)) * 4
    vertical_faces = vertical_faces.reshape((-1, 3))
    # 将 (n,2) 顶点与零堆叠以使其成为 (n, 3)
    vertices_3D = util.three_dimensionalize(vertices, return_2D=False)
    # 零索引的面序列,然后将其附加偏移量以创建最终网格
    faces_seq = [faces[:, ::-1], faces.copy(), vertical_faces]
    vertices_seq = [vertices_3D, (vertices_3D.copy() + [0.0, 0, height]), vertical]
    mesh = Trimesh(*util.append_faces(vertices_seq, faces_seq), process=True)
    return mesh


def triangulate_polygon(polygon, **kwargs):
    '''
    给定一个 Shapely 多边形,使用 meshpy.triangle 创建三角剖分

    :param polygon: Shapely.geometry.Polygon 对象
    :param kwargs: 直接传递给 meshpy.triangle.build 的参数
    :return mesh_vertices: (n, 2) float 数组,表示二维点
    :return mesh_faces: (n, 3) int 数组,表示三角形的顶点索引
    '''
    # 在这里进行导入,因为有时这个导入会导致 Python 崩溃
    import meshpy.triangle as triangle

    def round_trip(start, length):
        '''
        给定起始索引和长度,创建一系列 (n, 2) 边以创建闭合遍历

        示例:
        start, length = 0, 3
        返回:[(0,1), (1,2), (2,0)]
        '''
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        # coords 是多边形边界上的 (n, 2) 有序点列表
        # 第一个和最后一个点相同,并且没有保证点不重复
        coords = np.array(boundary.coords)
        # 找到仅出现一次的点的索引,并对其进行排序以保持顺序
        unique = np.sort(unique_rows(coords)[0])
        cleaned = coords[unique]

        vertices.append(cleaned)
        facets.append(round_trip(start, len(cleaned)))

        # 孔需要在孔区域内的点,我们通过从清理后的边界区域创建多边形来找到它
        # 然后使用代表点.您可以做一些事情,比如取点的平均值,但这更稳健
        test = Polygon(cleaned)
        holes.append(np.array(test.representative_point().coords)[0])
        return len(cleaned)

    # 空间中(n,2)个点的序列
    vertices = deque()
    # 顶点索引(n,2)序列
    facets = deque()
    # 空洞区域内部的(2)顶点列表
    holes = deque()

    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try:
            start += add_boundary(interior, start)
        except:
            log.warn('无效内部,继续')
            continue

    # 创建一个干净的(n,2)浮点数组 and (m, 2) int面片数组通过堆叠(p,2)数组序列
    vertices = np.vstack(vertices)
    facets = np.vstack(facets)
    # 在meshpy术语中,洞是由(x,y)点组成的(h, 2)列表
    # 它们都在洞的区域内
    # 我们为外部添加了一个洞,并在这里切掉
    holes = np.array(holes)[1:]
    # 调用meshpy.三角形在我们清洁的形状多边形表示
    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(facets)
    info.set_holes(holes)

    # 使用kwargs
    mesh = triangle.build(info, **kwargs)

    mesh_vertices = np.array(mesh.points)
    mesh_faces = np.array(mesh.elements)

    return mesh_vertices, mesh_faces


def box():
    '''
    返回一个单位立方体,中心在原点,边长为 1.0
    '''
    vertices = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    vertices = np.array(vertices, dtype=np.float64).reshape((-1, 3))
    vertices -= 0.5
    faces = [1, 3, 0, 4, 1, 0, 0, 3, 2, 2, 4, 0, 1, 7, 3, 5, 1, 4,
             5, 7, 1, 3, 7, 2, 6, 4, 2, 2, 7, 6, 6, 5, 4, 7, 5, 6]
    faces = np.array(faces, dtype=np.int64).reshape((-1, 3))
    face_normals = [-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 1, 0, -1,
                    0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    face_normals = np.array(face_normals, dtype=np.float64).reshape(-1, 3)
    box = Trimesh(vertices=vertices,
                  faces=faces,
                  face_normals=face_normals)
    return box


def icosahedron():
    '''
    创建一个二十面体
    '''
    t = (1.0 + 5.0 ** .5) / 2.0
    v = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
         0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]
    f = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
         1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
         3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
         4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
    v = np.reshape(v, (-1, 3))
    f = np.reshape(f, (-1, 3))
    m = Trimesh(v, f)
    return m


def icosphere(subdivisions=3):
    '''
    创建一个以原点为中心的半径为 1.0 的等距球体

    :param subdivisions: int,细分的次数
    :return ico: Trimesh 对象,表示等距球体
    '''

    def refine_spherical():
        vectors = ico.vertices
        scalar = (vectors ** 2).sum(axis=1) ** .5
        unit = vectors / scalar.reshape((-1, 1))
        offset = 1.0 - scalar
        ico.vertices += unit * offset.reshape((-1, 1))

    ico = icosahedron()
    ico._validate = False
    for j in range(subdivisions):
        ico.subdivide()
        refine_spherical()
    ico._validate = True
    return ico


def uv_sphere(radius=1.0, count=[32, 32], theta=None, phi=None):
    """
    创建一个以原点为中心的 UV 球体(纬度和经度)  比 icosphere 快一个数量级,但稍微不那么美观

    :param radius: 球体的半径
    :param count: 纬度和经度线的数量
    :param theta: 可选的 theta 角度(弧度)
    :param phi: 可选的 phi 角度(弧度)
    :return: Trimesh,具有指定参数的 UV 球体的网格
    """
    count = np.array(count, dtype=int)
    count += np.mod(count, 2)
    count[1] *= 2
    # 生成球体上的顶点,使用球面坐标
    if theta is None:
        theta = np.linspace(0, np.pi, count[0])
    if phi is None:
        phi = np.linspace(0, np.pi * 2, count[1])[:-1]
    spherical = np.dstack((np.tile(phi, (len(theta), 1)).T,
                           np.tile(theta, (len(phi), 1)))).reshape((-1, 2))
    vertices = util.spherical_to_vector(spherical) * radius
    # 生成面,通过创建一系列扇形
    c = len(theta)
    # 一个四边形面表示两个三角形
    pairs = np.array([[c, 0, 1], [c + 1, c, 1]])
    # 将每个四边形面的两个三角形增加相同的偏移量
    incrementor = np.tile(np.arange(c - 1), (2, 1)).T.reshape((-1, 1))
    # 为球面创建一个饼状的楔形面
    strip = np.tile(pairs, (c - 1, 1))
    strip += incrementor
    # 第一张face和最后一张face将比第一张脸退化 和最后一个顶点在两行中相同
    strip = strip[1:-1]
    # 瓷砖饼向球体楔入
    faces = np.vstack([strip + (i * c) for i in range(len(phi))])
    # 极点在每个条带中重复,所以用mask来合并它们
    mask = np.arange(len(vertices))
    # 顶极点都是相同的顶点
    mask[0::c] = 0
    # 底部的极点都是相同的顶点
    mask[c - 1::c] = c - 1
    # 屏蔽面以删除重复的顶点 并对最后一个饼形块进行模组填充
    faces = mask[np.mod(faces, len(vertices))]
    # 不用再次处理,我们节省了很多时间,因为我们做了一些簿记网格是防水的
    mesh = Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def cylinder(height, radius, sections=8, homomat=None):
    """
    创建一个沿 Z 轴从原点开始的圆柱体网格

    :param height: 圆柱体的高度
    :param radius: 圆柱体的半径
    :param sections: 圆柱体被网格化为多少个扇形
    :param homomat: 4x4 变换矩阵
    :return: Trimesh,生成的网格

    author: weiwei
    date: 20191228
    """
    theta = np.linspace(0, np.pi * 2, sections)
    vertices = np.column_stack((np.sin(theta), np.cos(theta))) * radius
    vertices[0] = [0, 0]
    index = np.arange(1, len(vertices) + 1).reshape((-1, 1))
    index[-1] = 1
    faces = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    faces = np.column_stack((np.zeros(len(faces), dtype=int), faces))
    cylinder = extrude_triangulation(vertices=vertices,
                                     faces=faces,
                                     height=height)
    # 将下面这行代码切换到原点居中位置
    # cylinder.vertices[:,2] -= height * .5
    if homomat is not None:
        cylinder.apply_transform(homomat)
    return cylinder


def capsule(height=1.0, radius=1.0, count=[8, 8], homomat=None):
    """
    创建一个胶囊体网格,或带有半球形末端的圆柱体

    :param radius: 圆柱体和半球的半径
    :param height: 圆柱体的高度,不包括半球
    :param count: 半球的网格化细节
    :param homomat: 4x4 变换矩阵
    :return: Trimesh,生成的胶囊体网格

    author: weiwei
    date: 20191228
    """
    height = float(height)
    radius = float(radius)
    count = np.array(count, dtype=int)
    count += np.mod(count, 2)
    # 在赤道周围有一个双带的地方创建一个theta 这样我们就可以偏移球体的顶部和底部获得一个漂亮的网格胶囊
    theta = np.linspace(0, np.pi, count[0])
    center = np.clip(np.arctan(tol.merge / radius), tol.merge, np.inf)
    offset = np.array([-center, center]) + (np.pi / 2)
    theta = np.insert(theta, int(len(theta) / 2), offset)
    capsule = uv_sphere(radius=radius, count=count, theta=theta)
    top = capsule.vertices[:, 2] > tol.zero
    capsule.vertices[top] += [0, 0, height]
    if homomat is not None:
        capsule.apply_transform(homomat)
    return capsule


def cone(height=1.0, radius=1.0, sections=8, homomat=None):
    """
    创建一个圆锥体网格

    :param height: float,圆锥体的高度
    :param radius: float,圆锥体底部的半径
    :param sections: int,圆锥体底部的分段数量
    :param homomat: np.ndarray,变换矩阵(可选)
    :return: Trimesh,表示圆锥体的网格对象

    author: weiwei
    date: 20191228
    """
    # 创建一个圆形的底部
    theta = np.linspace(0, np.pi * 2, sections)
    vertices = np.column_stack((np.sin(theta), np.cos(theta), np.zeros(sections, dtype=int))) * radius
    vertices[0] = [0, 0, 0]
    index = np.arange(1, len(vertices) + 1).reshape((-1, 1))
    index[-1] = 1
    bottomfaces = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    bottomfaces = np.column_stack((np.zeros(len(bottomfaces), dtype=int), bottomfaces))

    # 创建帽
    vertices = np.vstack((vertices, [0, 0, height]))
    index = np.arange(1, len(vertices)).reshape((-1, 1))
    index[-1] = 1
    capfaces = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    capfaces = np.column_stack((np.full(len(capfaces), sections, dtype=int), capfaces[:, 1], capfaces[:, 0]))
    faces = np.vstack((bottomfaces, capfaces))
    cone = Trimesh(vertices=vertices, faces=faces)
    if homomat is not None:
        cone.apply_transform(homomat)
    return cone
