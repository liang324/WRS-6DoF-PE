'''
github.com/mikedh/trimesh

Library for importing, exporting and doing simple operations on triangular meshes.
'''

import numpy as np
from copy import deepcopy
from . import triangles
from . import grouping
from . import geometry
from . import graph
from . import visual
from . import sample
from . import repair
from . import comparison
from . import boolean
from . import intersections
from . import util
from . import convex
from . import remesh
from . import bounds
from . import units
from . import inertia
from .io.export import export_mesh
from .ray.ray_mesh import RayMeshIntersector, contains_points
from .voxel import Voxel
from .points import transform_points
from .constants import log, _log_time, tol
from .scene import Scene

try:
    from .path.io.misc import faces_to_path
    from .path.io.load import _create_path, load_path
except ImportError:
    log.warning('trimesh.path unavailable, try pip install shapely!',
                exc_info=True)


class Trimesh(object):
    def __init__(self, vertices=None, faces=None, face_normals=None, vertex_normals=None, metadata=None, process=False,
                 **kwargs):
        """
        Trimesh 对象包含一个三角形 3D 网格

        :param vertices: nx3 的 numpy 数组,表示顶点坐标
        :param faces: mx3 的 numpy 数组,表示三角形面
        :param face_normals: mx3 的 numpy 数组,表示面法线
        :param vertex_normals: nx3 的 numpy 数组,表示顶点法线
        :param metadata: 字典,包含网格的元数据
        :param process: 布尔值,如果为 True,则在实例化时进行基本网格清理
        :param kwargs: 其他参数

        author: revised by weiwei
        date: 20201201
        """
        # self._data 存储关于网格的信息,这些信息无法再生.
        # 在基类中,存储在这里的只是顶点和面信息.
        # 任何放入存储的数据都会被转换为 TrackedArray(np.ndarray 的子类)
        # 提供了一个 md5() 方法,可以用于检测数组中的变化.
        self._data = util.DataStore()
        # self._cache 存储关于网格的信息,这些信息可以从 self._data 再生,但可能计算较慢.
        # 为了保持一致性,当 self._data.md5() 发生变化时,缓存会被清除.
        self._cache = util.Cache(id_function=self._data.md5)
        # 仅检查 None 以避免在子类中出现警告消息
        if vertices is not None:
            # (n, 3) 浮点数,顶点集合
            self.vertices = vertices
        if faces is not None:
            # (m, 3) 整数的三角形面,引用 self.vertices
            self.faces = faces
        # 保存关于网格的视觉信息(顶点和面颜色)
        # if 'visual' in kwargs:
        #     self.visual = kwargs['visual']
        # else:
        self.visual = visual.VisualAttributes(**kwargs)
        self.visual.mesh = self
        # 法线通过 setter/property 访问,如果维度不一致则会重新生成,
        # 但可以通过构造函数设置,以节省生成它们所需的大量叉乘.
        self.face_normals = face_normals
        # (n, 3) 浮点数的顶点法线,可以从面法线创建
        self.vertex_normals = vertex_normals
        # 为当前网格创建一个射线-网格查询对象
        # 初始化非常便宜,且对象使用起来很方便.
        # 在第一次查询时进行昂贵的簿记(创建 r-tree),
        # 并为后续查询缓存.
        self.ray = RayMeshIntersector(self)
        # 在字典中存储关于网格的元数据
        self.metadata = dict()
        # 使用传递的元数据更新网格元数据
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        # 在返回面或面法线时,验证面以确保非零法线和匹配的形状.
        # 不验证可能意味着根据查看面和面法线的顺序,您会得到不同数量的值,
        # 但对于某些操作,可能希望在操作期间关闭验证,然后在操作结束时重新初始化.
        self._validate = True
        # process 是一个清理函数,通过合并顶点和删除零面积和重复面来使网格达到一致状态.
        if ((process) and (vertices is not None) and (faces is not None)):
            self.process()

    def process(self):
        """
        方便函数,用于去除垃圾并使网格正常

        通过以下方式实现:
            1) 合并重复顶点
            2) 删除重复面
            3) 删除零面积面
        author: Revised by weiwei
        date: 20201201
        """
        # 如果没有顶点或面,提前退出
        if self.is_empty:
            return self
        # 在操作期间避免清除缓存
        with self._cache:
            self.merge_vertices()
            self.remove_duplicate_faces()
            self.remove_degenerate_faces()
        # 由于我们的处理操作没有移动顶点或面,
        # 我们可以在不重新计算的情况下将面和顶点法线保留在缓存中.
        # 如果面或顶点已被删除,法线在返回前会被验证,
        # 因此不存在维度不一致的危险.
        self._cache.clear(exclude=['face_normals',
                                   'vertex_normals'])
        self.metadata['processed'] = True
        return self

    @property
    def faces(self):
        """
        网格的面.
        这被视为无法从缓存再生的核心信息,
        因此存储在 self._data 中,该数据跟踪数组的变化,
        并在此被更改时清除网格的缓存值.
        :return faces: nx3 的 numpy 数组,表示引用 self.vertices 的三角形
        author: Revised by weiwei
        date: 20201201
        """
        # 我们验证面法线,因为验证过程可能会删除零面积面.
        # 如果我们不在这里进行检查,self.faces 和 self.face_normals 的形状可能会根据查询顺序而有所不同.
        self._validate_face_normals()
        return self._data['faces']

    @faces.setter
    def faces(self, values):
        """
        设置网格的面
        如果输入为四边形面,则进行三角化处理

        :param values: 面的索引数组,可以是三角形或四边形
        """
        if values is None: values = []
        values = np.asanyarray(values, dtype=np.int64)
        if util.is_shape(values, (-1, 4)):
            log.info('四边形面进行三角化处理')
            values = geometry.triangulate_quads(values)
        self._data['faces'] = values

    @property
    def faces_sparse(self):
        """
        面的稀疏矩阵表示

        :return sparse: scipy.sparse.coo_matrix,dtype=bool,形状为 (len(self.vertices), len(self.faces))

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['faces_sparse']
        if cached is not None:
            return cached
        sparse = geometry.index_sparse(column_count=len(self.vertices), indices=self.faces)
        self._cache['faces_sparse'] = sparse
        return sparse

    @property
    def face_normals(self):
        """
        获取面法线
        """
        self._validate_face_normals()
        cached = self._cache['face_normals']
        return cached

    @face_normals.setter
    def face_normals(self, values):
        """
        设置面法线

        :param values: 面法线数组
        """
        self._cache['face_normals'] = np.asanyarray(values)

    @property
    def vertices(self):
        """
        网格的顶点

        这被视为无法从缓存再生的核心信息
        因此存储在 self._data 中,该数据跟踪数组的变化
        并在此被更改时清除网格的缓存值
        :return vertices: (n,3) 浮点数,表示笛卡尔空间中的点

        author: Revised by weiwei
        date: 20201201
        """
        return self._data['vertices']

    @vertices.setter
    def vertices(self, values):
        """
        设置网格的顶点

        确保顶点以 float64 存储以保持一致性

        :param values: 顶点坐标数组
        """
        self._data['vertices'] = np.asanyarray(values, dtype=np.float64)

    def _validate_face_normals(self, faces=None):
        """
        确保面法线的形状正确
        此函数还会删除零面积的面,因此在返回面或三角形之前必须调用此函数
        以避免根据函数调用顺序导致结果不一致

        :param faces: nx3 的 numpy 数组,如果为 None,则使用 self.faces
               作为参数以避免某些函数中的循环引用

        author: Revised by weiwei
        date: 20201201
        """
        if not self._validate:
            return
        # 直接从 DataStore 中提取面以避免无限递归
        if faces is None:
            faces = self._data['faces']
        if np.shape(self._cache.get('face_normals')) != np.shape(faces):
            log.debug('生成面法线,因为形状不正确')
            tri_cached = self.vertices.view(np.ndarray)[faces]
            face_normals, valid = triangles.normals(tri_cached)
            self.update_faces(valid)
            self._cache['face_normals'] = face_normals

    @property
    def vertex_normals(self):
        """
        获取网格的顶点法线.如果法线已加载,我们会检查顶点法线和顶点的数量是否相同,然后再返回它们
        如果没有定义顶点法线,或者形状不匹配,我们会根据顶点所使用的面的平均法线来计算顶点法线

        :return vertex_normals: nx3 的浮点数组,其中 n == len(self.vertices),表示每个顶点的表面法线

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['vertex_normals']
        if np.shape(cached) == np.shape(self.vertices):
            return cached
        vertex_normals = geometry.mean_vertex_normals(len(self.vertices), self.faces, self.face_normals,
                                                      sparse=self.faces_sparse)
        self._cache['vertex_normals'] = vertex_normals
        return vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, values):
        """
        设置顶点法线

        :param values: 顶点法线数组
        """
        self._cache['vertex_normals'] = np.asanyarray(values)

    def md5(self):
        """
        获取网格核心几何信息(面和顶点)的 MD5 值
        由 TrackedArray 生成,该类继承自 np.ndarray 以监控更改,并返回正确但延迟计算的 md5(因此只需偶尔重新计算哈希,而不是每次调用时)

        :return md5: 字符串,面和顶点的 numpy 数组的附加 md5 哈希值

        author: Revised by weiwei
        date: 20201201
        """
        md5 = self._data.md5()
        return md5

    @property
    def bounding_box(self):
        """
        获取当前网格的轴对齐边界框

        :return aabb: trimesh.primitives.Box 对象,具有定义的变换和范围,以表示网格的轴对齐边界框.

        author: Revised by weiwei
        date: 20201201
        """
        aabb = self._cache['aabb']
        if aabb is None:
            from . import primitives
            aabb = primitives.Box(box_center=self.bounds.mean(axis=0),
                                  box_extents=self.extents)
            self._cache['aabb'] = aabb
        return aabb

    @property
    def bounding_box_oriented(self):
        """
        获取当前网格的有向边界框

        :return obb: trimesh.primitives.Box 对象,具有定义的变换和范围,以表示网格的最小体积有向边界框

        author: Revised by weiwei
        date: 20201201
        """
        obb = self._cache['obb']
        if obb is None:
            from . import primitives
            to_origin, extents = bounds.oriented_bounds(self)
            obb = primitives.Box(box_transform=np.linalg.inv(to_origin), box_extents=extents)
            self._cache['obb'] = obb
        return obb

    @property
    def bounds(self):
        """
        获取网格的轴对齐边界

        :return bounds: 2x3 的浮点数组,表示边界框的 [最小值, 最大值] 坐标

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['bounds']
        if cached is not None:
            return cached
        # 我们使用三角形而不是面,因为如果有未使用的顶点,它会影响边界
        in_mesh = self.triangles.reshape((-1, 3))
        bounds = np.vstack((in_mesh.min(axis=0), in_mesh.max(axis=0)))
        self._cache['bounds'] = bounds
        return bounds

    @property
    def extents(self):
        """
        网格边界框的长度、宽度和高度

        :return extents: 1x3 的浮点数组,包含轴对齐的 [长度, 宽度, 高度]

        author: Revised by weiwei
        date: 20201201
        """
        extents = np.diff(self.bounds, axis=0)[0]
        return extents

    @property
    def scale(self):
        """
        网格整体尺度的度量

        :return scale: float,网格边界框的最长边

        author: Revised by weiwei
        date: 20201201
        """
        scale = self.extents.max()
        return scale

    @property
    def centroid(self):
        """
        网格的平均顶点所在的空间点

        :return centroid: 1x3 的浮点数,表示平均顶点

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['centroid']
        if cached is not None:
            return cached
        # 使用三角形而不是顶点,因为顶点可能包含未使用的点
        in_mesh = self.triangles.reshape((-1, 3))
        centroid = in_mesh.mean(axis=0)
        self._cache['centroid'] = centroid
        return centroid

    @property
    def center_mass(self):
        """
        网格的质心/体积中心所在的空间点
        如果当前网格不是封闭的,这将是无意义的数据

        :return center_mass: (3,) 的浮点数组,网格的体积质心

        author: Revised by weiwei
        date: 20201201
        """
        center_mass = np.array(self.mass_properties(skip_inertia=True)['center_mass'])
        return center_mass

    @property
    def volume(self):
        """
        当前网格的体积
        如果当前网格不是封闭的,这将是无意义的数据

        :return volume: float,当前网格的体积

        author: Revised by weiwei
        date: 20201201
        """
        volume = self.mass_properties(skip_inertia=True)['volume']
        return volume

    @property
    def moment_inertia(self):
        """
        返回当前网格的惯性矩阵
        如果网格不是封闭的,这将是无意义的数据

        :return inertia: 1x3 的 nparray,当前网格的惯性矩

        author: Revised by weiwei
        date: 20201201
        """
        inertia = np.array(self.mass_properties(skip_inertia=False)['inertia'])
        return inertia

    @property
    def triangles(self):
        """
        网格的实际三角形(点,不是索引)

        :return triangles: nx3x3 的 nparray,顶点按三角形分组

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['triangles']
        if cached is not None:
            return cached
        # 使用高级索引会触发更改标志,这意味着需要重新计算 MD5
        # 我们可以通过查看数组来避免此检查
        triangles = self.vertices.view(np.ndarray)[self.faces]
        # 使得从面/顶点派生的三角形不可写
        triangles.flags.writeable = False
        self._cache['triangles'] = triangles
        return triangles

    def triangles_tree(self):
        """
        包含网格每个面的 R 树

        :return tree: rtree.index,其中 self.faces 中的每个三角形都有一个矩形单元

        author: Revised by weiwei
        date: 20201201
        """
        tree = triangles.bounds_tree(self.triangles)
        return tree

    @property
    def edges(self):
        """
        网格的边(从面派生)

        :return edges: nx2 的 int nparray,顶点索引集

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['edges']
        if cached is not None:
            return cached
        edges, index = geometry.faces_to_edges(self.faces.view(np.ndarray), return_index=True)
        self._cache['edges'] = edges
        self._cache['edges_face'] = index
        return edges

    @property
    def edges_face(self):
        """
        每条边属于哪个面

        :return edges_face: 1xn 的 int nparray,self.faces 的索引

        author: revised by weiwei
        date: 20201201
        """
        populate = self.edges
        return self._cache['edges_face']

    @property
    def edges_unique(self):
        """
        网格的唯一边

        :return edges_unique: nx2 的 int 数组,唯一边的顶点索引集

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['edges_unique']
        if cached is not None: return cached
        unique, inverse = grouping.unique_rows(self.edges_sorted)
        edges_unique = self.edges_sorted[unique]
        self._cache['edges_unique'] = edges_unique
        self._cache['edges_unique_idx'] = unique
        self._cache['edges_unique_inv'] = inverse
        return edges_unique

    @property
    def faces_unique_edges(self):
        """
        对于每个面,返回构成该面的 mesh.unique_edges 的索引

        :return faces_unique_edges: self.faces.shape int,构成 self.faces 的 self.edges_unique 的索引

        :example
        In [0]: mesh.faces[0:2]
        Out[0]:
        TrackedArray([[    1,  6946, 24224],
                      [ 6946,  1727, 24225]])
        In [1]: mesh.edges_unique[mesh.faces_unique_edges[0:2]]
        Out[1]:
        array([[[    1,  6946],
                [ 6946, 24224],
                [    1, 24224]],
               [[ 1727,  6946],
                [ 1727, 24225],
                [ 6946, 24225]]])
        """
        # 确保我们已填充唯一边
        populate = self.edges_unique
        # 我们依赖于边按三元组堆叠的事实
        result = self._cache['edges_unique_inv'].reshape((-1, 3))
        return result

    @property
    def edges_sorted(self):
        """
        返回排序后的边(沿轴1排序)

        :return: self.edges,排序后的边

        author: Revised by weiwei
        date: 20201201
        """
        cached = self._cache['edges_sorted']
        if cached is not None:
            return cached
        edges_sorted = np.sort(self.edges, axis=1)
        self._cache['edges_sorted'] = edges_sorted
        return edges_sorted

    @property
    def euler_number(self):
        """
        返回网格的欧拉特征(拓扑不变量)
        为了保证正确性,应在调用 remove_unreferenced_vertices 之后调用此函数.

        :return: euler_number,int,拓扑不变量

        author: Revised by weiwei
        date: 20201201
        """
        euler = len(self.vertices) - len(self.edges_unique) + len(self.faces)
        return euler

    @property
    def units(self):
        """
        网格的单位定义

        :return: units,str,网格的单位系统,如果未定义则为 None

        author: Revised by weiwei
        date: 20201201
        """
        return self.metadata['units'] if 'units' in self.metadata else None

    @units.setter
    def units(self, value):
        """
        设置网格的单位

        :param value: str,新的单位
        :raises ValueError: 如果单位无效
        """
        value = str(value)
        if not units.validate(value):
            raise ValueError(value + ' 不是有效的单位!')
        self.metadata['units'] = value

    def convert_units(self, desired, guess=False):
        """
        将网格的单位转换为指定单位

        :param desired: str,目标单位(例如 'inches')
        :param guess: bool,如果 self.units 未定义,是否猜测当前文档的单位然后进行转换
        :return: None

        author: Revised by weiwei
        date: 20201201
        """
        units._set_units(self, desired, guess)

    def merge_vertices(self, angle=None):
        """
        如果网格有顶点距离小于 TOL_MERGE,将它们重新定义为相同的顶点,并替换面引用

        :param angle: 如果定义,仅合并距离小于 TOL_MERGE 且顶点法线小于 angle_max 的顶点.
                      这对于平滑着色很有用,但速度较慢.
        :return: None

        author: Revised by weiwei
        date: 20201201
        """
        if angle is None:
            grouping.merge_vertices_hash(self)
        else:
            grouping.merge_vertices_kdtree(self, angle)

    def update_vertices(self, mask, inverse=None):
        """
        更新顶点

        :param mask: 1xlen(self.vertices) 的布尔数组,表示要保留的顶点
        :param inverse: 1xlen(self.vertices) 的 int 数组,用于重建顶点引用(例如由 np.unique 输出)
        :return: None

        author: Revised by weiwei
        date: 20201201
        """
        mask = np.asanyarray(mask)
        if mask.dtype.name == 'bool' and mask.all():
            return
        if len(mask) == 0 or self.is_empty:
            return
        if inverse is not None:
            self.faces = inverse[np.array(self.faces.reshape(-1))].reshape((-1, 3))
        self.visual.update_vertices(mask)
        cached_normals = self._cache.get('vertex_normals')
        if util.is_shape(cached_normals, (-1, 3)):
            try:
                self.vertex_normals = cached_normals[mask]
            except:
                pass
        self.vertices = self.vertices[mask]

    def update_faces(self, mask):
        """
        更新面

        在许多情况下,我们需要移除特定的面
        然而,为了干净地做到这一点,还有额外的记录工作
        此函数使用有效性掩码更新面集,并跟踪法线和颜色

        :param mask: (m) int 或 (len(self.faces)) bool
        :return: None
        """
        if self.is_empty:
            return
        mask = np.asanyarray(mask)
        if mask.dtype.name == 'bool':
            if mask.all():
                return
        elif mask.dtype.name != 'int':
            mask = mask.astype(int)
        cached_normals = self._cache.get('face_normals')
        if util.is_shape(cached_normals, (-1, 3)):
            self.face_normals = cached_normals[mask]
            # except: pass
        faces = self._data['faces']
        # 是否对Trimesh进行了子类化并且 faces已经从数据中移动了 为了缓存,从缓存中获取 faces
        if not util.is_shape(faces, (-1, 3)):
            faces = self._cache['faces']
        self.faces = faces[mask]
        self.visual.update_faces(mask)

    def remove_duplicate_faces(self):
        """
        从当前网格中移除任何重复的面
        :return: None
        """
        unique, inverse = grouping.unique_rows(np.sort(self.faces, axis=1))
        self.update_faces(unique)

    def rezero(self):
        """
        平移网格,使所有顶点坐标为正
        """
        self.apply_translation(self.bounds[0] * -1.0)

    @_log_time
    def split(self, only_watertight=True, adjacency=None):
        """
        根据面连接性返回 Trimesh 对象的列表
        将网格分割为独立的组件,有时称为“实体”

        :param only_watertight: 仅返回水密的网格
        :param adjacency: 如果不为 None,用自定义值覆盖面邻接关系 (n,2)
        :return: Trimesh 对象的列表
        """
        meshes = graph.split(self, only_watertight=only_watertight, adjacency=adjacency)
        log.info('split found %i components', len(meshes))
        return meshes

    @property
    def face_adjacency(self):
        """
        查找共享边的面,这里称为“相邻”

        :return: adjacency: (n,2) int,面对的对,它们共享一条边
        :example:
        In [1]: mesh = trimesh.load('models/featuretype.STL')
        In [2]: mesh.face_adjacency
        Out[2]:
        array([[   0,    1],
               [   2,    3],
               [   0,    3],
               ...,
               [1112,  949],
               [3467, 3475],
               [1113, 3475]])
        In [3]: mesh.faces[mesh.face_adjacency[0]]
        Out[3]:
        TrackedArray([[   1,    0,  408],
                      [1239,    0,    1]], dtype=int64)
        In [4]: import networkx as nx
        In [5]: graph = nx.from_edgelist(mesh.face_adjacency)
        In [6]: groups = nx.connected_components(graph)
        """
        cached = self._cache['face_adjacency']
        if cached is not None:
            return cached
        adjacency, edges = graph.face_adjacency(self.faces.view(np.ndarray), return_edges=True)
        self._cache['face_adjacency_edges'] = edges
        self._cache['face_adjacency'] = adjacency
        return adjacency

    @property
    def face_adjacency_edges(self):
        """
        返回由相邻面共享的边

        :return: edges: (n, 2) 顶点索引列表,对应于 face_adjacency
        """
        cached = self._cache['face_adjacency_edges']
        if cached is not None:
            return cached
        # 这个值是面邻接的副产品
        populate = self.face_adjacency
        return self._cache['face_adjacency_edges']

    @property
    def is_winding_consistent(self):
        '''
        网格的绕线是否一致
        具有一致绕线的网格,每个共享边在对中与另一个方向相反

        :return: consistent: bool,绕线是否一致
        '''
        # 一致绕线检查通过 is_watertight 查询填充到缓存中
        populate = self.is_watertight
        return self._cache['is_winding_consistent']

    @property
    def is_watertight(self):
        '''
        通过确保每条边被两个面使用来检查网格是否水密

        :return: is_watertight: bool,网格是否水密
        '''
        cached = self._cache.get('is_watertight')
        if cached is not None:
            return cached
        watertight, reversed = graph.is_watertight(self.edges, return_winding=True)
        self._cache['is_watertight'] = watertight
        self._cache['is_winding_consistent'] = reversed
        return watertight

    @property
    def is_empty(self):
        '''
        检查当前网格是否有定义的数据

        :returns empty: 如果为 True,则网格中不存在数据
        '''
        return self._data.is_empty()

    @property
    def is_convex(self):
        '''
        检查网格是否为凸形

        返回值   is_convex: bool,网格是否为凸形
        '''

        cached = self._cache['is_convex']
        if cached is not None:
            return cached
        is_convex = convex.is_convex(self)
        self._cache['is_convex'] = is_convex
        return is_convex

    def kdtree(self):
        '''
        返回网格顶点的 scipy.spatial.cKDTree
        不缓存,因为这会导致内存问题和段错误

        返回值
        ---------
        tree: scipy.spatial.cKDTree,包含网格顶点
        '''
        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(self.vertices.view(np.ndarray))
        return tree

    def remove_degenerate_faces(self):
        '''
        从当前网格中移除退化面(没有3个唯一顶点索引的面)
        '''
        nondegenerate = triangles.nondegenerate(self.triangles)
        self.update_faces(nondegenerate)

    def facets(self, return_area=False):
        '''
        返回共面相邻面的面索引列表

        参数
        ---------
        return_area: boolean,如果为 True,则返回每组面的面积

        返回值
        ---------
        facets: (n) 面索引序列
        area:   (n) 面组面积的浮点列表(如果 return_area)
        '''
        key = 'facets_' + str(int(return_area))
        cached = self._cache[key]
        if cached is not None:
            return cached

        facets = graph.facets(self)
        if return_area:
            area = np.array([self.area_faces[i].sum() for i in facets])
            result = (facets, area)
        else:
            result = facets
        self._cache[key] = result
        return result

    def facets_boundary(self):
        """
        返回表示每个面边界的边

        返回值
        ---------
        edges_boundary : (n, 2) int 序列   self.vertices 的索引
        """
        # 让每一行对应一个面
        edges = self.edges_sorted.reshape((-1, 6))
        # 获取每个面的边
        edges_facet = [edges[i].reshape((-1, 2)) for i in self.facets]
        edges_boundary = [i[grouping.group_rows(i, require_count=1)] for i in edges_facet]
        return edges_boundary

    # def facets_depth(self, return_area=False):
    #     return graph.facets_depth(self)

    def facets_over(self, face_angle=.9, seg_angle=.9):
        """
        使用过度分割计算面

        参数
        ---------
        face_angle: 两个相邻面被认为是共面的角度
        seg_angle: 分割角度

        返回值
        ---------
        facets: 分割后的面列表及其法线

        author: weiwei
        date: 20161128tsukuba
        """
        key = 'facets_over'
        cached = self._cache[key]
        if cached is not None:
            return cached
        facets, facetnormals, curvatures = graph.facets_over_segmentation(self, face_angle, seg_angle)
        result = [facets, facetnormals, curvatures]
        self._cache[key] = result
        return result

    def facets_noover(self, faceangle=.9):
        """
        使用非重叠分割计算面,此函数用于比较

        :param faceangle: 两个相邻面被认为是共面的角度
        :return facets: 分割后的面列表及其法线

        author: weiwei
        date: 20190806tsukuba
        """
        key = 'facets_noover'
        cached = self._cache[key]
        if cached is not None:
            return cached
        facets, facetnormals, curvatures = graph.facets_noover(self, faceangle)
        result = [facets, facetnormals, curvatures]
        self._cache[key] = result
        return result

    @_log_time
    def fix_normals(self):
        """
        找到并修复 self.face_normals 和 self.faces 绕线方向的问题
        确保面法线向量一致指向外部,并且 self.faces 在所有连接组件中绕线正确
        """
        repair.fix_normals(self)

    def fill_holes(self):
        """
        填充当前网格中的单三角形和单四边形孔

        :return: watertight: bool, 函数完成后网格是否水密
        """
        return repair.fill_holes(self)

    def subdivide(self, face_index=None):
        """
        细分网格,每个细分面替换为四个较小的面

        :param: face_index: 要细分的面
                            如果为 None:网格的所有面都将被细分
                            如果为 (n,) int 索引数组:仅指定的面将被细分
                            注意,在这种情况下,网格通常不再是流形,因为中点上的额外顶点不会被相邻面使用
                            需要额外的后处理步骤来使结果网格水密
        :return: mesh: Trimesh 对象
        """
        remesh.subdivide(self, face_index=face_index)

    @_log_time
    def smoothed(self, angle=.4):
        """
        返回当前网格的版本,该版本将很好地渲染.不会以任何方式改变当前网格
        :return: smoothed: Trimesh 对象,当前网格的非水密版本,将以平滑着色很好地渲染
        """
        # 如果视觉效果发生变化,smooth应该重新计算,所以我们将其存储在视觉缓存中,而不是主mesh缓存中
        cached = self.visual._cache.get('smoothed')
        if cached is not None:
            return cached
        return self.visual._cache.set(key='smoothed', value=graph.smoothed(self, angle))

    def section(self, plane_normal, plane_origin=None):
        """
        返回当前网格与由原点和法线定义的平面的截面

        :param plane_normal: 1x3 nparray,平面法线
        :param plane_origin: 1x2 向量,平面原点
        :return: intersections: Path3D,交点路径
        """
        lines = intersections.mesh_plane(mesh=self, plane_normal=plane_normal, plane_origin=plane_origin)
        if len(lines) == 0:
            raise ValueError('指定的平面不与网格相交！')
        path = load_path(lines)
        return path

    @property
    def _convex_hull_raw(self):
        '''
        从 qhull 返回的原始凸包

        return: hull: Trimesh 对象,来自 qhull 的原始凸包,面向后
        '''
        cached = self._cache['convex_hull_raw']
        if cached is not None:
            return cached
        hull = convex.convex_hull(self, clean=False)
        self._cache['convex_hull_raw'] = hull
        return hull

    @property
    def convex_hull(self):
        '''
        获取表示当前网格凸包的新 Trimesh 对象.需要 scipy >.12

        :return convex: 当前网格凸包的 Trimesh 对象
        '''
        cached = self._cache.get('convex_hull')
        if cached is not None:
            return cached
        hull = self._convex_hull_raw.copy()
        hull.fix_normals()
        return self._cache.set(key='convex_hull', value=hull)

    def sample_surface(self, count, radius=None, toggle_faceid=False):
        """
        返回随机样本,通常分布在网格表面上

        :param count: int,采样点的数量
        :param radius: float,采样半径(可选)
        :param toggle_faceid: bool,是否返回相关面 ID
        :return: samples: countx3 浮点数组,网格表面上的点；faceids: 1xcount 列表(如果 toggle_faceid 为 True)

        author: revised by weiwei
        date: 20201202toyonaka
        """
        if radius is not None:
            points, index = sample.sample_surface_even(mesh=self, count=count, radius=radius)
        else:
            points, index = sample.sample_surface(mesh=self, count=count)
        if toggle_faceid:
            return points, index
        return points

    def remove_unreferenced_vertices(self):
        '''
        移除当前网格中所有未被面引用的顶点
        '''
        unique, inverse = np.unique(self.faces.reshape(-1), return_inverse=True)
        self.faces = inverse.reshape((-1, 3))
        self.vertices = self.vertices[unique]

    def unmerge_vertices(self):
        '''
        移除所有面引用,使每个面包含三个唯一的顶点索引,并且没有面是相邻的
        '''
        with self._cache:
            self.update_vertices(mask=self.faces.reshape(-1))
            self.faces = np.arange(len(self.vertices)).reshape((-1, 3))
        self._cache.clear(exclude='face_normals')

    def apply_translation(self, translation):
        '''
        平移当前网格

        :param translation: (3,) 浮点数组,XYZ方向的平移量
        '''
        translation = np.asanyarray(translation).reshape(3)
        with self._cache:
            self.vertices += translation
        # 我们正在进行简单的平移,因此法线保持不变
        self._cache.clear(exclude=['face_normals',
                                   'vertex_normals'])

    def apply_scale(self, scaling):
        """
        应用缩放变换

        :param scaling: [scale_x, scale_y, scale_z],缩放比例

        author: weiwei
        date: 20210403
        """
        assert (scaling[0] > 0 and scaling[1] > 0 and scaling[2] > 0)
        matrix = np.diag([scaling[0], scaling[1], scaling[2], 1])
        self.apply_transform(matrix)

    def apply_transform(self, matrix):
        """
        通过齐次变换矩阵变换网格,同时变换法线以避免重新计算

        :param matrix: 4x4 齐次变换矩阵
        :return: self,变换后的网格对象

        author: weiwei
        date: 20210414
        """
        matrix = np.asanyarray(matrix)
        if matrix.shape != (4, 4):
            raise ValueError('变换矩阵必须为 (4,4)!')
        new_vertices = transform_points(self.vertices, matrix)
        new_normals = None
        if self.faces is not None:
            new_normals = np.dot(matrix[0:3, 0:3], self.face_normals.T).T
            # 比计算矩阵的比例因子更简单
            new_normals = util.unitize(new_normals)
            # 检查第一个面与第一个法线是否正确缠绕
            aligned_pre = triangles.windings_aligned(self.vertices[self.faces[:1]],
                                                     self.face_normals[:1])[0]
            aligned_post = triangles.windings_aligned(new_vertices[self.faces[:1]],
                                                      new_normals[:1])[0]
            if aligned_pre != aligned_post:
                log.debug('变换后三角形法线未对齐；翻转')
                self.faces = np.fliplr(self.faces)
        with self._cache:
            self.vertices = new_vertices
            self.face_normals = new_normals
        self._cache.clear(exclude=['face_normals'])
        log.debug('网格通过矩阵变换,法线恢复到缓存')
        return self

    def voxelized(self, pitch):
        '''
        返回一个 Voxel 对象,表示当前网格被离散化为指定间距的体素

        :param pitch: float,单个体素的边长
        :return: voxelized,表示当前网格的 Voxel 对象
        '''
        voxelized = Voxel(self, pitch)
        return voxelized

    def outline(self, face_ids=None):
        '''
        给定一组面 ID,找到面的轮廓,并将其作为 Path3D 返回
        轮廓在此定义为仅由一个三角形包含的每个边
        注意,这意味着非水密部分,水密网格的“轮廓”是空路径

        :param face_ids: (n) int,self.faces 的索引列表,用于计算轮廓
                         如果为 None,将计算整个网格的轮廓
        :return: path,轮廓的 Path3D 对象
        '''
        path = _create_path(**faces_to_path(self, face_ids))
        return path

    @property
    def area(self):
        '''
        计算当前网格中所有三角形的总面积

        :return area: float,网格的表面积
        '''
        cached = self._cache['area']
        if cached is not None:
            return cached
        area = self.area_faces.sum()
        self._cache['area'] = area
        return area

    @property
    def area_faces(self):
        '''
        计算网格中每个面的面积

        :return area_faces: (n,) float,每个面的面积
        '''
        cached = self._cache['area_faces']
        if cached is not None:
            return cached
        area_faces = triangles.area(self.triangles, sum=False)
        self._cache['area_faces'] = area_faces
        return area_faces

    def mass_properties(self, density=1.0, skip_inertia=False):
        '''
        返回当前网格的质量属性
        假设密度均匀,如果网格不是封闭的,结果可能不准确

        :param density: float,固体的密度
        :param skip_inertia: bool,是否跳过惯性计算
        :return properties: dict,包含以下键:
                            'volume'      : 体积,单位为全局单位的立方
                            'mass'        : 从指定密度计算的质量
                            'density'     : 为方便起见再次包含的密度(与参数 density 相同)
                            'inertia'     : 在质心处并与全局坐标系对齐的惯性
                            'center_mass' : 质心位置,单位为全局坐标系
        '''
        key = 'mass_properties_'
        key += str(int(skip_inertia)) + '_'
        key += str(int(density * 1e5))
        cached = self._cache[key]
        if cached is not None:
            return cached
        mass = triangles.mass_properties(triangles=self.triangles,
                                         density=density,
                                         skip_inertia=skip_inertia)
        self._cache[key] = mass
        return mass

    def scene(self):
        """
        获取包含当前网格的 Scene 对象

        :return: trimesh.scene.scene.Scene 对象,包含当前网格
        """
        return Scene(self)

    def show(self, block=True, **kwargs):
        """
        在 OpenGL 窗口中渲染网格.需要 pyglet

        :param block: bool,是否在新线程中打开窗口,或阻塞直到窗口关闭
        :param smooth: bool,是否对网格运行平滑着色.大网格会很慢
        :return scene: trimesh.scene.Scene 对象,包含当前网格的场景
        """
        scene = self.scene()
        scene.show(block=block, **kwargs)
        return scene

    def submesh(self, faces_sequence, **kwargs):
        """
        获取网格的子集

        :param faces_sequence: 从网格中选择的面索引序列
        :param only_watertight: 仅返回封闭的子网格
        :param append: 返回一个包含指定面的单一网格.如果设置此标志,则忽略 only_watertight
        :return: 如果 append 为 True,则返回 Trimesh 对象,否则返回 Trimesh 对象的列表
        """
        return util.submesh(mesh=self, faces_sequence=faces_sequence, **kwargs)

    @property
    def identifier(self):
        """
        返回一个对网格唯一且对旋转和平移具有鲁棒性的浮点向量

        :return identifier: (tol.id_len,) float
        """
        key = 'identifier'
        cached = self._cache[key]
        if cached is not None:
            return cached
        identifier = comparison.rotationally_invariant_identifier(self, tol.id_len)
        self._cache[key] = identifier
        return identifier

    def export(self, file_obj=None, file_type='meshes'):
        """
        将当前网格导出到文件对象
        如果 file_obj 是文件名,文件将写入该位置
        支持的格式有 meshes、off 和 collada
        """
        return export_mesh(mesh=self, file_obj=file_obj, file_type=file_type)

    def to_dict(self):
        """
        返回当前网格的字典表示形式,键可以用作 Trimesh 构造函数的关键字参数,
        例如:a = Trimesh(**other_mesh.to_dict())

        :return result: dict,与 trimesh 构造函数匹配的键
        """
        result = self.export(file_type='dict')
        return result

    def union(self, other, engine=None):
        """
        当前网格与其他网格的布尔并集

        :param other: Trimesh 对象或 Trimesh 对象的列表
        :return union: Trimesh,self 和其他 Trimesh 对象的并集
        """
        return Trimesh(process=True, **boolean.union(meshes=np.append(self, other), engine=engine))

    def difference(self, other, engine=None):
        """
        当前网格与其他网格的布尔差集

        :param other: Trimesh 对象或 Trimesh 对象的列表
        :return difference: Trimesh,self 和其他 Trimesh 对象的差集
        """
        return Trimesh(process=True, **boolean.difference(meshes=np.append(self, other), engine=engine))

    def intersection(self, other, engine=None):
        """
        当前网格与其他网格的布尔交集

        :param other: Trimesh 对象或 Trimesh 对象的列表
        :return intersection: Trimesh,所有传入网格包含的体积
        """
        return Trimesh(process=True, **boolean.intersection(meshes=np.append(self, other), engine=engine))

    def contains(self, points):
        """
        给定一组点,确定它们是否在网格内部
        如果在非封闭网格上调用此方法,将引发错误

        :param points: nx3 空间中的点集
        :return contains: (n) 布尔数组,点是否在网格内
        """
        if not self.is_watertight:
            log.warning('网格不是封闭的,无法进行包含点查询！')
        contains = contains_points(self, points)
        return contains

    def copy(self):
        """
        获取当前网格的副本

        :return copied: 当前网格的深拷贝
        """
        return deepcopy(self)

    def __hash__(self):
        # 哈希函数需要一个整数而不是十六进制字符串
        hashed = int(self.md5(), 16)
        return hashed

    def __add__(self, other):
        """
        将网格与另一个网格连接

        author: revised by weiwei
        date: 20210120
        """
        result = util.concatenate(self, other)
        return result

    @property
    def symmetry(self):
        """
        检查网格是否具有围绕轴(径向)或点(球形)的旋转对称性

        :return  symmetry: None, 'radial', 'spherical'网格具有哪种对称性.
        """
        cached = self._cache.get('symmetry')
        if cached is not None:
            return cached
        symmetry, axis, section = inertia.radial_symmetry(self)
        self._cache['symmetry_axis'] = axis
        self._cache['symmetry_section'] = section
        return symmetry

    @property
    def symmetry_axis(self):
        """
        如果网格具有旋转对称性,返回轴

        :returns axis : (3, ) float 围绕其旋转 2D 轮廓以创建此网格的轴
        """
        if self.symmetry is not None:
            return self._cache['symmetry_axis']

    @property
    def symmetry_section(self):
        """
        如果网格具有旋转对称性,返回构成截面坐标系的两个向量

        :return  section : (2, 3) float 用于截面的向量
        """
        if self.symmetry is not None:
            return self._cache['symmetry_section']

    @property
    def principal_inertia_components(self):
        """
        返回惯性主成分
        顺序对应于 mesh.principal_inertia_vectors

        :return components : (3, ) float惯性主成分
        """
        cached = self._cache.get('principal_inertia_components')
        if cached is not None:
            return cached
        # 从惯性矩阵中获取成分和向量
        components, vectors = inertia.principal_axis(self.moment_inertia)
        # 将向量存储在缓存中以备后用
        self._cache['principal_inertia_vectors'] = vectors
        self._cache['principal_inertia_components'] = components
        return components

    @property
    def principal_inertia_vectors(self):
        """
        以单位向量形式返回惯性主轴
        顺序对应于 mesh.principal_inertia_components

        :return  vectors : (3, 3) float 指向惯性主轴方向的三个向量
        """
        cached = self._cache.get('principal_inertia_vectors')
        if cached is not None:
            return cached
        populate = self.principal_inertia_components
        return self._cache['principal_inertia_vectors']
