import numpy as np
from . import util
from . import transformations


def sample_surface(mesh, count):
    """
    从网格的表面采样,返回指定数量的点

    对于单个三角形采样使用此方法:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    :param mesh: 一个 Trimesh 实例
    :param count: 要返回的点的数量
    :return: 返回采样的点和对应的面索引

    author: revised by weiwei
    date: 20200120
    """
    # len(mesh.faces) 浮点数组,表示网格中每个面的面积
    area = mesh.area_faces
    # 累积面积(len(mesh.faces))
    area_sum = np.sum(area)
    # 累积面积(len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)
    # 将三角形转换为原点 + 2 个向量的形式
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    # 提取我们要采样的面的向量
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    # 随机生成两个 0-1 的标量分量,用于乘以边缘向量
    random_lengths = np.random.random((len(tri_vectors), 2, 1))
    # 如果使用两个 0-1 的样本,点将在四边形上分布
    # 如果两个标量分量的和小于 1.0,点将在三角形内部
    # 因此我们找到和大于 1.0 的向量,并将它们转换为三角形内部
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    # 用随机长度乘以三角形边缘向量并求和
    points_vector = (tri_vectors * random_lengths).sum(axis=1)
    # 最后,通过原点偏移生成空间中的 (n,3) 点在三角形上
    points = points_vector + tri_origins
    return points, face_index


# def sample_surface_withfaceid(mesh, count):
#     '''
#     Sample the surface of a mesh, returning the specified number of points
#
#     For individual triangle sampling uses this method:
#     http://mathworld.wolfram.com/TrianglePointPicking.html
#
#     Arguments
#     ---------
#     mesh: Trimesh objects
#     count: number of points to return
#
#     Returns
#     ---------
#     samples: (count,3) points in space on the surface of mesh
#
#     '''
#
#     # len(mesh.faces) float array of the areas of each face of the mesh
#     area = mesh.area_faces
#     # total area (float)
#     area_sum = np.sum(area)
#     # cumulative area (len(mesh.faces))
#     area_cum = np.cumsum(area)
#     face_pick = np.random.random(count) * area_sum
#     face_index = np.searchsorted(area_cum, face_pick)
#
#     # pull triangles into the form of an origin + 2 vectors
#     tri_origins = mesh.triangles[:, 0]
#     tri_vectors = mesh.triangles[:, 1:].copy()
#     tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
#
#     # pull the vectors for the faces we are going to sample from
#     tri_origins = tri_origins[face_index]
#     tri_vectors = tri_vectors[face_index]
#
#     # randomly generate two 0-1 scalar components to multiply edge vectors by
#     random_lengths = np.random.random((len(tri_vectors), 2, 1))
#
#     # points will be distributed on a quadrilateral if we use 2 0-1 samples
#     # if the two scalar components sum less than 1.0 the point will be
#     # inside the triangle, so we find vectors longer than 1.0 and
#     # transform them to be inside the triangle
#     random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
#     random_lengths[random_test] -= 1.0
#     random_lengths = np.abs(random_lengths)
#
#     # multiply triangle edge vectors by the random lengths and sum
#     sample_vector = (tri_vectors * random_lengths).sum(axis=1)
#
#     # finally, offset by the origin to generate
#     # (n,3) points in space on the triangle
#     samples = sample_vector + tri_origins
#
#     return samples, face_index


def sample_volume(mesh, count):
    """
    使用拒绝采样在网格的体积中随机分布生成点

    :param mesh: 网格对象
    :param count: 要生成的点的数量
    :return: 在网格体积内的点

    author: revised by weiwei
    date: 20210120
    """
    points = (np.random.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples


def sample_box_volume(extents,
                      count,
                      transform=None):
    """
    使用拒绝采样在给定盒子的体积中随机分布生成点

    :param extents: 1x3 的 numpy 数组,表示盒子的尺寸
    :param count: 要生成的点的数量
    :param transform: 齐次变换矩阵
    :return: 请求体积内的 nx3 点

    author: revised by weiwei
    date: 20210120
    """
    samples = np.random.random((count, 3)) - .5
    samples *= extents
    if transform is not None:
        samples = transformations.transform_points(samples, transform)
    return samples


def sample_surface_even(mesh, count, radius=None):
    """
    从网格的表面采样,返回大致均匀分布的样本

    请注意,由于使用拒绝采样,可能返回的点数少于请求的数量(即 n < count)如果出现这种情况,将发出 log.warning

    :param mesh: 网格对象
    :param count: 要生成的点的数量
    :param radius: 采样点之间的最小距离
    :return: 采样点和对应的面索引

    author: revised by weiwei
    date: 20210120
    """
    from .points import remove_close_withfaceid
    # 从面积猜测半径
    if radius is None:
        radius = np.sqrt(mesh.area / (3 * count))
    # 获取表面上的点
    points, index = sample_surface(mesh, count * 3)
    # 移除距离小于半径的点
    points, index = remove_close_withfaceid(points, index, radius)
    # 如果获得了所有期望的样本
    if len(points) >= count:
        return points[:count], index[:count]
    # 如果没有获得所有期望的样本,发出警告
    # util.log.warning('only got {}/{} samples!'.format(len(points), count)) TODO
    return points, index


def sample_surface_sphere(count):
    """
    正确地在单位球的表面上随机选择点

    使用此方法:http://mathworld.wolfram.com/SpherePointPicking.html

    :param count: 要生成的点的数量
    :return: 单位球上的 nx3 点
    """
    # 获取随机值 0.0-1.0
    u, v = np.random.random((2, count))
    # 转换为两个角度
    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)
    # 将球面坐标转换为笛卡尔坐标
    points = util.spherical_to_vector(np.column_stack((theta, phi)))
    return points
