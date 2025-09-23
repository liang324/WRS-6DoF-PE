import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from scipy.spatial import cKDTree as KDTree
from copy import deepcopy
from collections import deque
from trimesh.points import plane_fit
from .simplify import simplify_path, points_to_spline_entity
from .polygons import polygons_enclosure_tree, medial_axis, polygon_hash, path_to_polygon, polygon_obb
from .traversal import vertex_graph, closed_paths, discretize_path
from .io.export import export_path
from ..points import transform_points
from ..geometry import plane_transform
from ..grouping import unique_rows
from ..units import _set_units
from ..util import decimal_to_digits
from ..constants import log
from ..constants import tol_path as tol
from .. import util


class Path(object):
    '''
    Path 对象由两部分组成

    vertices: (n,[2|3]) 坐标,存储在 self.vertices 中
    entities: 几何图元(线、弧和圆),引用 self.vertices 中的索引
    '''

    def __init__(self,
                 entities=[],
                 vertices=[],
                 metadata=None):
        '''
        初始化 Path 对象

        :param entities: 包含关键点等内容的对象,作为对 self.vertices 的引用
        :param vertices: (n, (2|3)) 顶点列表
        :param metadata: 可选的元数据字典
        '''
        self.entities = np.array(entities)
        self.vertices = vertices
        self.metadata = dict()

        if metadata.__class__.__name__ == 'dict':
            self.metadata.update(metadata)

        self._cache = util.Cache(id_function=self.md5)
        # 如果顶点没有正确合并,几乎没有任何东西会正常工作
        self.merge_vertices()

    def process(self):
        # 处理绘图
        log.debug('Processing drawing')
        with self._cache:
            for func in self._process_functions():
                func()
        return self

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, values):
        self._vertices = util.tracked_array(values)

    def md5(self):
        # 计算对象的 MD5 哈希值
        result = self.vertices.md5()
        result += str(len(self.entities))
        return result

    @property
    def paths(self):
        # 获取缓存中的路径,如果不存在则计算并缓存
        if 'paths' in self._cache:
            return self._cache.get('paths')
        with self._cache:
            paths = closed_paths(self.entities, self.vertices)
        return self._cache.set('paths', paths)

    @property
    def kdtree(self):
        # 获取或创建 KDTree 对象
        if 'kdtree' in self._cache:
            return self._cache.get('kdtree')
        with self._cache:
            kdtree = KDTree(self.vertices.view(np.ndarray))
        return self._cache.set('kdtree', kdtree)

    @property
    def scale(self):
        # 计算顶点的最大范围
        return np.max(np.ptp(self.vertices, axis=0))

    @property
    def bounds(self):
        # 计算顶点的边界框
        return np.vstack((np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)))

    @property
    def extents(self):
        # 计算边界框的范围
        return np.diff(self.bounds, axis=0)[0]

    @property
    def units(self):
        # 获取单位信息
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @property
    def is_closed(self):
        # 检查路径是否闭合
        return all(i == 2 for i in self.vertex_graph.degree().values())

    @property
    def vertex_graph(self):
        # 获取或创建顶点图
        if 'vertex_graph' in self._cache:
            return self._cache.get('vertex_graph')
        with self._cache:
            graph, closed = vertex_graph(self.entities)
        return self._cache.set('vertex_graph', graph)

    @units.setter
    def units(self, units):
        # 设置单位信息
        self.metadata['units'] = units

    def set_units(self, desired, guess=False):
        # 设置单位
        _set_units(self, desired, guess)

    def transform(self, transform):
        # 应用变换到顶点
        self.vertices = transform_points(self.vertices, transform)

    def rezero(self):
        # 将顶点重新定位到原点
        self.vertices -= self.vertices.min(axis=0)

    def merge_vertices(self):
        '''
        合并相同的顶点并替换引用
        '''
        digits = decimal_to_digits(tol.merge * self.scale, min_digits=1)
        unique, inverse = unique_rows(self.vertices, digits=digits)
        self.vertices = self.vertices[unique]
        for entity in self.entities:
            entity.points = inverse[entity.points]

    def replace_vertex_references(self, replacement_dict):
        # 替换顶点引用
        for entity in self.entities: entity.rereference(replacement_dict)

    def remove_entities(self, entity_ids):
        # 根据索引移除实体
        if len(entity_ids) == 0: return
        kept = np.setdiff1d(np.arange(len(self.entities)), entity_ids)
        self.entities = np.array(self.entities)[kept]

    def remove_invalid(self):
        # 移除无效实体
        valid = np.array([i.is_valid for i in self.entities], dtype=bool)
        self.entities = self.entities[valid]

    def remove_duplicate_entities(self):
        # 移除重复实体
        entity_hashes = np.array([i.hash for i in self.entities])
        unique, inverse = unique_rows(entity_hashes)
        if len(unique) != len(self.entities):
            self.entities = np.array(self.entities)[unique]

    def referenced_vertices(self):
        """
        获取所有被引用的顶点

        :return: 被引用的顶点数组
        """
        # 获取所有被引用的顶点
        referenced = deque()
        for entity in self.entities:
            referenced.extend(entity.points)
        return np.array(referenced)

    def remove_unreferenced_vertices(self):
        """
        移除所有未被实体引用的顶点

        将顶点重新从零开始索引,并替换引用
        """
        referenced = self.referenced_vertices()
        unique_ref = np.int_(np.unique(referenced))
        replacement_dict = dict()
        replacement_dict.update(np.column_stack((unique_ref, np.arange(len(unique_ref)))))
        self.replace_vertex_references(replacement_dict)
        self.vertices = self.vertices[[unique_ref]]

    def discretize_path(self, path):
        '''
        返回一个 (n, dimension) 的顶点列表

        将弧/曲线采样为线段

        :param path: 要离散化的路径
        :return: 离散化后的顶点列表
        '''
        discrete = discretize_path(self.entities,
                                   self.vertices,
                                   path,
                                   scale=self.scale)
        return discrete

    def paths_to_splines(self, path_indexes=None, smooth=.0002):
        '''
        将路径转换为 B 样条

        :param path_indexes: (n) int 列表,表示 self.paths 的索引
        :param smooth: float,样条平滑曲线的程度
        '''
        if path_indexes is None:
            path_indexes = np.arange(len(self.paths))
        entities_keep = np.ones(len(self.entities), dtype=bool)
        new_vertices = deque()
        new_entities = deque()
        for i in path_indexes:
            path = self.paths[i]
            discrete = self.discrete[i]
            entity, vertices = points_to_spline_entity(discrete)
            entity.points += len(self.vertices) + len(new_vertices)
            new_vertices.extend(vertices)
            new_entities.append(entity)
            entities_keep[path] = False
        self.entities = np.append(self.entities[entities_keep], new_entities)
        self.vertices = np.vstack((self.vertices, np.array(new_vertices)))

    def export(self, file_type='dict', file_obj=None):
        """
        导出路径数据

        :param file_type: 导出的文件类型
        :param file_obj: 文件对象
        :return: 导出的路径数据
        """
        return export_path(self, file_type=file_type, file_obj=file_obj)

    def to_dict(self):
        """
        将路径转换为字典格式

        :return: 路径的字典表示
        """
        export_dict = self.export(file_type='dict')
        return export_dict

    def copy(self):
        """
        创建路径的深拷贝

        :return: 路径的副本
        """
        return deepcopy(self)

    def show(self):
        # 显示路径
        if self.is_closed:
            self.plot_discrete(show=True)
        else:
            self.plot_entities(show=True)

    def __add__(self, other):
        """
        合并两个路径

        :param other: 另一个路径对象
        :return: 合并后的新路径
        """
        new_entities = deepcopy(other.entities)
        for entity in new_entities:
            entity.points += len(self.vertices)
        new_entities = np.append(deepcopy(self.entities), new_entities)
        new_vertices = np.vstack((self.vertices, other.vertices))
        new_meta = deepcopy(self.metadata)
        new_meta.update(other.metadata)
        new_path = self.__class__(entities=new_entities, vertices=new_vertices, metadata=new_meta)
        return new_path


class Path3D(Path):
    def _process_functions(self):
        """
        返回处理函数列表

        :return: 处理函数列表
        """
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices,
                self.generate_closed_paths,
                self.generate_discrete]

    @property
    def discrete(self):
        """
        获取离散化的路径

        :return: 离散化的路径列表
        """
        cached = self._cache['discrete']
        if cached is not None:
            return cached
        discrete = list(map(self.discretize_path, self.paths))
        self._cache['discrete'] = discrete
        return discrete

    def to_planar(self, to_2D=None, normal=None, check=True):
        """
        检查当前向量是否共面
        如果是,返回一个 Path2D 和一个变换,该变换将2D 表示转换回三维

        :param to_2D: 2D 转换矩阵
        :param normal: 法向量
        :param check: 是否检查共面性
        :return: Path2D 对象和 3D 转换矩阵
        """
        if to_2D is None:
            C, N = plane_fit(self.vertices)
            if normal is not None:
                N *= np.sign(np.dot(N, normal))
            to_2D = plane_transform(C, N)

        flat = transform_points(self.vertices, to_2D)
        if check and np.any(np.std(flat[:, 2]) > tol.planar):
            log.error('点的 z 偏差为 %f', np.std(flat[:, 2]))
            raise NameError('Points aren\'t planar!')

        vector = Path2D(entities=deepcopy(self.entities), vertices=flat[:, 0:2])
        to_3D = np.linalg.inv(to_2D)
        return vector, to_3D

    def plot_discrete(self, show=False):
        """
        绘制离散化的路径

        :param show: 是否显示图形
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for discrete in self.discrete:
            axis.plot(*discrete.T)
        if show: plt.show()

    def plot_entities(self, show=False):
        """
        绘制路径中的实体

        :param show: 是否显示图形
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for entity in self.entities:
            vertices = self.vertices[entity.points]
            axis.plot(*vertices.T)
        if show: plt.show()


class Path2D(Path):
    def _process_functions(self):
        """
        返回处理函数列表

        :return: 处理函数列表
        """
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices]

    def apply_obb(self):
        """
        应用定向边界框(OBB)变换
        """
        if len(self.root) == 1:
            bounds, T = polygon_obb(self.polygons_closed[self.root[0]])
            self.transform(T)
        else:
            raise ValueError('不支持多体几何体')

    @property
    def body_count(self):
        """
        获取主体数量

        :return: 主体数量
        """
        return len(self.root)

    @property
    def polygons_full(self):
        """
        获取完整的多边形列表

        :return: 完整的多边形列表
        """
        if 'polygons_full' in self._cache:
            return self._cache.get('polygons_full')

        with self._cache:
            result = [None] * len(self.root)
            for index, root in enumerate(self.root):
                hole_index = self.connected_paths(root, include_self=False)
                holes = [p.exterior.coords for p in self.polygons_closed[hole_index]]
                shell = self.polygons_closed[root].exterior.coords
                result[index] = Polygon(shell=shell, holes=holes)
        return self._cache.set('polygons_full', result)

    @property
    def area(self):
        """
        返回多边形内部的面积

        :return: 多边形的面积
        """
        area = np.sum([i.area for i in self.polygons_full])
        return area

    def extrude(self, height, **kwargs):
        """
        将当前 2D 路径拉伸为 3D 网格

        :param height: 拉伸的高度
        :param kwargs: 传递给 meshpy.triangle.build 的其他参数
        :return: 表示拉伸多边形的 trimesh 对象
        """
        from ..primitives import Extrusion
        result = [Extrusion(polygon=i, height=height, **kwargs) for i in self.polygons_full]
        if len(result) == 1:
            return result[0]
        return result

    def medial_axis(self, resolution=None, clip=None):
        """
        基于多边形边界上均匀分布点的 Voronoi 图,找到近似的中轴线

        :param resolution: 多边形边界上每个样本之间的目标距离
        :param clip: [最小样本数, 最大样本数],指定非常细的分辨率可能导致样本数爆炸,
                     因此 clip 指定每个边界区域要使用的最小和最大样本数.
                     如果不进行裁剪,可以指定为:[0, np.inf]
        :return: 中轴线的 Path2D 对象
        """
        if 'medial' in self._cache:
            return self._cache.get('medial')
        if resolution is None:
            resolution = self.scale / 1000.0
        medials = [medial_axis(i, resolution, clip) for i in self.polygons_full]
        medials = np.sum(medials)
        return self._cache.set(key='medial', value=medials)

    def connected_paths(self, path_id, include_self=False):
        """
        获取与给定路径 ID 连接的路径 ID 列表

        :param path_id: 要检查的路径 ID
        :param include_self: 是否包含自身路径 ID
        :return: 连接的路径 ID 数组
        """
        if len(self.root) == 1:
            path_ids = np.arange(len(self.polygons_closed))
        else:
            path_ids = list(nx.node_connected_component(self.enclosure, path_id))
        if include_self:
            return np.array(path_ids)
        return np.setdiff1d(path_ids, [path_id])

    def simplify(self):
        """
        简化路径,清除缓存并调用简化函数
        """
        self._cache.clear()
        simplify_path(self)

    def split(self):
        """
        如果当前 Path2D 由 n 个 'root' 曲线组成,将其拆分为 n 个 Path2D 对象的列表

        :return: 拆分后的 Path2D 对象数组
        """
        if self.root is None or len(self.root) == 0:
            split = []
        elif len(self.root) == 1:
            split = [deepcopy(self)]
        else:
            split = [None] * len(self.root)
            for i, root in enumerate(self.root):
                connected = self.connected_paths(root, include_self=True)
                new_root = np.nonzero(connected == root)[0]
                new_entities = deque()
                new_paths = deque()
                new_metadata = {'split_2D': i}
                new_metadata.update(self.metadata)

                for path in self.paths[connected]:
                    new_paths.append(np.arange(len(path)) + len(new_entities))
                    new_entities.extend(path)
                new_entities = np.array(new_entities)
                # 防止复制时清除缓存
                with self._cache:
                    split[i] = Path2D(entities=deepcopy(self.entities[new_entities]),
                                      vertices=deepcopy(self.vertices))
                    split[i]._cache.update({'paths': np.array(new_paths),
                                            'polygons_closed': self.polygons_closed[connected],
                                            'root': new_root})
        [i._cache.id_set() for i in split]
        self._cache.id_set()
        return np.array(split)

    def plot_discrete(self, show=False, transform=None, axes=None):
        """
        绘制离散化的路径

        :param show: 是否显示图形
        :param transform: 变换矩阵
        :param axes: 绘图轴
        """
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')

        def plot_transformed(vertices, color='g'):
            if transform is None:
                if axes is None:
                    plt.plot(*vertices.T, color=color)
                else:
                    axes.plot(*vertices.T, color=color)
            else:
                transformed = transform_points(vertices, transform)
                plt.plot(*transformed.T, color=color)

        for i, polygon in enumerate(self.polygons_closed):
            color = ['g', 'k'][i in self.root]
            plot_transformed(np.column_stack(polygon.boundary.xy), color=color)
        if show: plt.show()

    def plot_entities(self, show=False):
        """
        绘制路径中的实体

        :param show: 是否显示图形
        """
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        eformat = {'Line0': {'color': 'g', 'linewidth': 1},
                   'Line1': {'color': 'y', 'linewidth': 1},
                   'Arc0': {'color': 'r', 'linewidth': 1},
                   'Arc1': {'color': 'b', 'linewidth': 1},
                   'Bezier0': {'color': 'k', 'linewidth': 1},
                   'BSpline0': {'color': 'm', 'linewidth': 1},
                   'BSpline1': {'color': 'm', 'linewidth': 1}}
        for entity in self.entities:
            discrete = entity.discrete(self.vertices)
            e_key = entity.__class__.__name__ + str(int(entity.closed))
            plt.plot(discrete[:, 0],
                     discrete[:, 1],
                     **eformat[e_key])
        if show: plt.show()

    @property
    def identifier(self):
        """
        获取路径的唯一标识符,仅对单个主体有效

        :return: 路径的唯一标识符
        """
        if len(self.polygons_full) != 1:
            raise TypeError('Identifier only valid for single body')
        return polygon_hash(self.polygons_full[0])

    @property
    def polygons_valid(self):
        """
        获取有效的多边形

        :return: 有效的多边形
        """
        exists = self.polygons_closed
        return self._cache.get('polygons_valid')

    @property
    def discrete(self):
        """
        获取离散化的多边形

        :return: 离散化的多边形
        """
        if not 'discrete' in self._cache:
            test = self.polygons_closed
        return self._cache['discrete']

    @property
    def polygons_closed(self):
        """
        获取闭合的多边形

        :return: 闭合的多边形
        """
        if 'polygons_closed' in self._cache:
            return self._cache.get('polygons_closed')

        def reverse_path(path):
            for entity in self.entities[path]:
                entity.reverse()
            return path[::-1]

        with self._cache:
            discretized = [None] * len(self.paths)
            polygons = [None] * len(self.paths)
            valid = [False] * len(self.paths)
            for i, path in enumerate(self.paths):
                discrete = discretize_path(self.entities,
                                           self.vertices,
                                           path,
                                           scale=self.scale)
                candidate = path_to_polygon(discrete, scale=self.scale)
                if candidate is None:
                    continue
                if type(candidate).__name__ == 'MultiPolygon':
                    area_ok = np.array([i.area for i in candidate]) > tol.zero
                    if area_ok.sum() == 1:
                        candidate = candidate[np.nonzero(area_ok)[0][0]]
                    else:
                        continue
                if not candidate.exterior.is_ccw:
                    log.debug('Clockwise polygon detected, correcting!')
                    self.paths[i] = reverse_path(path)
                    candidate = Polygon(np.array(candidate.exterior.coords)[::-1])
                polygons[i] = candidate
                valid[i] = True
                discretized[i] = discrete
            valid = np.array(valid, dtype=bool)
            polygons = np.array(polygons)[valid]
            discretized = np.array(discretized)
        self._cache.set('discrete', discretized)
        self._cache.set('polygons_valid', valid)
        self._cache.set('polygons_closed', polygons)
        return polygons

    @property
    def root(self):
        """
        获取多边形的根节点

        :return: 根节点
        """
        if 'root' in self._cache:
            return self._cache.get('root')
        with self._cache:
            root, enclosure = polygons_enclosure_tree(self.polygons_closed)
        self._cache.set('enclosure_directed', enclosure)
        return self._cache.set('root', root)

    @property
    def enclosure(self):
        """
        获取多边形的无向包围树

        :return: 无向包围树
        """
        if 'enclosure' in self._cache:
            return self._cache.get('enclosure')
        with self._cache:
            undirected = self.enclosure_directed.to_undirected()
        return self._cache.set('enclosure', undirected)

    @property
    def enclosure_directed(self):
        """
        获取多边形的有向包围树

        :return: 有向包围树
        """
        if 'enclosure_directed' in self._cache:
            return self._cache.get('enclosure_directed')
        with self._cache:
            root, enclosure = polygons_enclosure_tree(self.polygons_closed)
        self._cache.set('root', root)
        return self._cache.set('enclosure_directed', enclosure)
