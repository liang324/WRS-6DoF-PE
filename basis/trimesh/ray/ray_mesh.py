import numpy as np

from ..util import Cache, unitize
from ..grouping import unique_rows
from ..intersections import plane_lines
from .ray_triangle_cpu import rays_triangles_id


class RayMeshIntersector:
    '''
    用于查询网格与射线相交的对象
    为网格上的每个三角形预计算一个 R 树
    '''

    def __init__(self, mesh):
        '''
        初始化 RayMeshIntersector 对象

        :param mesh: Mesh 对象,用于进行射线相交查询
        '''
        self.mesh = mesh
        self._cache = Cache(self.mesh.md5)

    @property
    def tree(self):
        '''
        获取或创建用于加速射线与三角形相交查询的 R 树
        '''
        if 'tree' in self._cache:
            return self._cache.get('tree')
        else:
            return self._cache.set('tree', self.mesh.triangles_tree())

    def intersects_id(self, rays, return_any=False):
        '''
        找到与射线相交的三角形的索引

        :param rays: (n, 2, 3) 数组,表示射线的起点和方向
        :param return_any: 布尔值,如果为 True,则在任意射线与三角形相交时提前退出循环
        :return: (n) 序列,表示与射线相交的三角形索引
        '''
        rays = np.array(rays, dtype=float)
        candidates = ray_triangle_candidates(rays=rays, tree=self.tree)
        hits = rays_triangles_id(triangles=self.mesh.triangles,
                                 rays=rays,
                                 ray_candidates=candidates,
                                 return_any=return_any)
        return hits

    def intersects_location(self, rays, return_id=False):
        '''
        返回射线与网格相交的唯一笛卡尔位置

        如果你在计算射线的命中次数,这个方法是否应该像只有三角形索引用于边缘命中一样使用将被计算两次

        :param rays: (n, 2, 3) 数组,表示射线的起点和方向
        :param return_id: 布尔标志,如果为 True,则返回三角形索引
        :return: (n) 序列,表示 (m,3) 的相交点；如果 return_id 为 True,还返回 (n) 的面 ID 列表
        '''
        rays = np.array(rays, dtype=float)
        hits = self.intersects_id(rays)
        locations = ray_triangle_locations(triangles=self.mesh.triangles,
                                           rays=rays,
                                           intersections=hits,
                                           tri_normals=self.mesh.face_normals)
        if return_id:
            return locations, hits
        return locations

    def intersects_any_triangle(self, rays):
        '''
        判断给定的射线是否与网格上的任意三角形相交

        :param rays: (n, 2, 3) 数组,表示射线的起点和方向
        :return: (n) 布尔数组,表示每条射线是否与任意三角形相交
        '''
        hits = self.intersects_id(rays)
        hits_any = np.array([len(i) > 0 for i in hits])
        return hits_any

    def intersects_any(self, rays):
        '''
        判断是否有任意射线与任意三角形相交

        :param rays: (n, 2, 3) 数组,表示射线的起点和方向
        :return: 布尔值,表示是否有任意射线与任意三角形相交
        '''
        hit = self.intersects_id(rays, return_any=True)
        return hit


def ray_triangle_candidates(rays, tree):
    '''
    对可能与射线相交的三角形进行广泛搜索
    通过为射线创建一个包围盒,使其穿过树所占据的体积来实现

    :param rays: 射线的数组
    :param tree: 用于加速查询的 R 树
    :return: 每条射线的候选三角形列表
    '''
    ray_bounding = ray_bounds(rays, tree.bounds)
    ray_candidates = [None] * len(rays)
    for ray_index, bounds in enumerate(ray_bounding):
        ray_candidates[ray_index] = list(tree.intersection(bounds))
    return ray_candidates


def ray_bounds(rays, bounds, buffer_dist=1e-5):
    '''
    给定一组射线和一个包围盒,计算射线穿过该区域时的包围盒

    :param rays: 射线的起点和方向,形状为 (n, 2, 3) 的数组
    :param bounds: 包围盒的最小值和最大值,形状为 (2, 3) 的数组
    :param buffer_dist: 零宽度包围盒的填充距离,默认为 1e-5
    :return: 射线穿过体积时的轴对齐包围盒,形状为 (n, 6) 的数组
    '''
    # 将 (n, 2, 3) 的射线数组分离为 (n, 3) 的起点和方向数组
    ray_ori = rays[:, 0, :]
    ray_dir = unitize(rays[:, 1, :])
    # 我们正在测试的包围盒
    bounds = np.array(bounds)
    # 找到向量的主要轴
    axis = np.abs(ray_dir).argmax(axis=1)
    axis_bound = bounds.reshape((2, -1)).T[axis]
    axis_ori = np.array([ray_ori[i][a] for i, a in enumerate(axis)]).reshape((-1, 1))
    axis_dir = np.array([ray_dir[i][a] for i, a in enumerate(axis)]).reshape((-1, 1))
    # 线的参数方程
    # point = direction*t + origin
    # p = dt + o
    # t = (p-o)/d
    t = (axis_bound - axis_ori) / axis_dir

    # 防止包围盒包含射线起点后面的三角形
    t[t < buffer_dist] = buffer_dist

    # 上下边界的 t 值
    t_a = t[:, 0].reshape((-1, 1))
    t_b = t[:, 1].reshape((-1, 1))

    # 线与平面相交的笛卡尔点
    on_a = (ray_dir * t_a) + ray_ori
    on_b = (ray_dir * t_b) + ray_ori

    on_plane = np.column_stack((on_a, on_b)).reshape((-1, 2, ray_dir.shape[1]))

    ray_bounding = np.hstack((on_plane.min(axis=1), on_plane.max(axis=1)))
    # 用 TOL_BUFFER 填充包围盒
    # 不确定这是否必要,但射线是否与轴对齐
    # 否则这个函数将返回0个体积边界框
    # 哪些可能会搞砸r树的交集查询
    ray_bounding += np.array([-1, -1, -1, 1, 1, 1]) * buffer_dist
    return ray_bounding


def ray_triangle_locations(triangles,
                           rays,
                           intersections,
                           tri_normals):
    '''
    给定一组三角形、射线和交点,计算交点的笛卡尔位置

    :param triangles: 三角形顶点,形状为 (n, 3, 3) 的数组
    :param rays: 射线的起点和方向,形状为 (m, 2, 3) 的数组
    :param intersections: 每条射线与哪个三角形相交的索引,序列
    :param tri_normals: 三角形的法向量,形状为 (n, 3) 的数组
    :return: 交点的笛卡尔坐标,形状为 (m, p, 3) 的数组
    '''
    ray_origin = rays[:, 0, :]
    ray_vector = rays[:, 1, :]
    ray_segments = np.array([ray_origin, ray_origin + ray_vector])
    locations = [[]] * len(rays)

    for r, tri_group in enumerate(intersections):
        group_locations = np.zeros((len(tri_group), 3))
        valid = np.zeros(len(tri_group), dtype=bool)
        for i, tri_index in enumerate(tri_group):
            origin = triangles[tri_index][0]
            normal = tri_normals[tri_index]
            segment = ray_segments[:, r, :].reshape((2, -1, 3))
            point, ok = plane_lines(plane_origin=origin,
                                    plane_normal=normal,
                                    endpoints=segment,
                                    line_segments=False)
            if ok:
                valid[i] = True
                group_locations[i] = point
        group_locations = group_locations[valid]
        unique = unique_rows(group_locations)[0]
        locations[r] = group_locations[unique]
    return np.array(locations)


def contains_points(mesh, points):
    '''
    使用射线测试检查网格是否包含一组点
    如果点在网格表面上,行为未定义

    :param mesh: Trimesh 对象
    :param points: 空间中的点,形状为 (n, 3) 的数组
    :return: 布尔数组,表示每个点是否在网格内,形状为 (n)
    '''
    points = np.asanyarray(points)
    vector = unitize([0, 0, 1])
    rays = np.column_stack((points, np.tile(vector, (len(points), 1)))).reshape((-1, 2, 3))
    hits = mesh.ray.intersects_location(rays)
    hits_count = np.array([len(i) for i in hits])
    contains = np.mod(hits_count, 2) == 1
    return contains
