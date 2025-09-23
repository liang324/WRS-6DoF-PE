import numpy as np
from .constants import tol, log
from .util import type_named, diagonal_dot
from .points import project_to_plane

try:
    from scipy.spatial import ConvexHull
except ImportError:
    log.warning('Scipy import failed!')


def convex_hull(mesh, clean=True):
    '''
    获取表示当前网格凸包的新 Trimesh 对象.需要 scipy > 0.12

    :param mesh: Trimesh 对象,表示输入网格
    :param clean: bool,如果为 True,将修复法线和绕组以保持一致
                  (因为 qhull/scipy 的输出不是一致的)

    :return convex: Trimesh 对象,表示当前网格的凸包
    '''
    type_trimesh = type_named(mesh, 'Trimesh')
    c = ConvexHull(mesh.vertices.view(np.ndarray).reshape((-1, 3)))

    vid = np.sort(c.vertices)
    mask = np.zeros(len(c.points), dtype=np.int64)
    mask[vid] = np.arange(len(vid))

    faces = mask[c.simplices]
    vertices = c.points[vid].copy()

    convex = type_trimesh(vertices=vertices,
                          faces=faces,
                          process=True)
    if clean:
        # scipy/qhull 的 ConvexHull 返回的法线和三角形绕组是随机的,因此我们需要完全修复它们
        convex.fix_normals()
    return convex


def is_convex(mesh, chunks=None):
    '''
    通过将三角形的顶点投影到其相邻面的法线上来测试网格是否为凸的

    :param mesh: Trimesh 对象,表示输入网格

    :return convex: bool,网格是否为凸的
    '''
    chunk_block = 5e4
    if chunks is None:
        chunks = int(np.clip(len(mesh.faces) / chunk_block, 1, 10))

    # 从面邻接的第二列中获取三角形
    triangles = mesh.triangles.copy()[mesh.face_adjacency[:, 1]]
    # 从面邻接的第一列中获取法线和原点
    normals = mesh.face_normals[mesh.face_adjacency[:, 0]]
    origins = mesh.vertices[mesh.face_adjacency_edges[:, 0]]

    # 重塑并平铺所有内容以使其具有相同的维度
    triangles = triangles.reshape((-1, 3))
    normals = np.tile(normals, (1, 3)).reshape((-1, 3))
    origins = np.tile(origins, (1, 3)).reshape((-1, 3))
    triangles -= origins

    # 在非凸网格中,我们不一定需要计算每个面的所有点积,因为我们正在寻找逻辑上的 ALL
    for chunk_tri, chunk_norm in zip(np.array_split(triangles, chunks),
                                     np.array_split(normals, chunks)):
        # 将相邻三角形的顶点投影到法线上
        # 注意,其中两个总是为零,所以我们实际上做的点积比我们真正需要的要多,
        # 但通过图形操作找到第三个顶点的索引比做额外的点积要慢得多.
        # 可能有一种巧妙的方法可以使用绕组来免费获得这个
        dots = diagonal_dot(chunk_tri, chunk_norm)
        # 如果所有投影都是负的,或者在三角形的“后面”,则网格是凸的
        if not bool((dots < tol.merge).all()):
            return False
    return True


def planar_hull(points, normal, origin=None, input_convex=False):
    '''
    找到投影到平面上的一组点的凸轮廓

    :param points: (n,3) float,输入点
    :param normal: (3) float 向量,平面的法向量
    :param origin: (3) float,平面原点的位置
    :param input_convex: bool,如果为 True,我们假设输入点已经来自凸包,
                         这提供了加速

    :return hull_lines: (n,2,2) 未排序的线段集合
    :return T: (4,4) float,变换矩阵
    '''
    if origin is None:
        origin = np.zeros(3)
    if not input_convex:
        pass
    planar, T = project_to_plane(points,
                                 plane_normal=normal,
                                 plane_origin=origin,
                                 return_planar=False,
                                 return_transform=True)
    hull_edges = ConvexHull(planar[:, 0:2]).simplices
    hull_lines = planar[hull_edges]
    planar_z = planar[:, 2]
    height = np.array([planar_z.min(),
                       planar_z.max()])
    return hull_lines, T, height


def hull_points(obj, qhull_options='QbB Pp'):
    """
    尝试从多种输入格式中提取凸点集

    :param obj: Trimesh 对象,(n,d) 点,(m,) Trimesh 对象
    :return points: (o,d) 凸点集
    """
    if hasattr(obj, 'convex_hull'):
        return obj.convex_hull.vertices

    initial = np.asanyarray(obj, dtype=np.float64)
    if len(initial.shape) != 2:
        raise ValueError('点必须是 (n, 维度)!')

    hull = ConvexHull(initial, qhull_options=qhull_options)
    points = hull.points[hull.vertices]

    return points
