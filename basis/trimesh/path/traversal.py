import numpy as np
import networkx as nx
from collections import deque
from ..grouping import unique_ordered
from ..util import unitize
from ..constants import tol_path as tol
from .util import is_ccw


def vertex_graph(entities):
    '''
    根据给定的实体对象集合(具有节点和闭合属性),生成一个顶点图

    :param entities: 实体对象的集合,每个对象具有节点和闭合属性
    :return: 一个包含顶点连接的图和一个表示闭合实体索引的数组
    '''
    graph = nx.Graph()
    closed = deque()
    for index, entity in enumerate(entities):
        if entity.closed:
            closed.append(index)
        else:
            graph.add_edges_from(entity.nodes, entity_index=index)
    return graph, np.array(closed)


def vertex_to_entity_path(vertex_path, graph, entities, vertices=None):
    '''
    将顶点索引路径转换为实体索引路径

    :param vertex_path: 顶点索引的列表
    :param graph: 顶点连接的图
    :param entities: 实体对象的列表
    :param vertices: 顶点的列表(可选)
    :return: 构成顶点路径的实体索引列表
    '''

    def edge_direction(a, b):
        '''
        给定两个边,确定第一个边是否需要反转以保持前进方向

        :param a: 第一个边的端点
        :param b: 第二个边的端点
        :return: 两个整数,表示边的方向

         [1,0] [1,2] -1  1
         [1,0] [2,1] -1 -1
         [0,1] [1,2]  1  1
         [0,1] [2,1]  1 -1
        '''
        if a[0] == b[0]:
            return -1, 1
        elif a[0] == b[1]:
            return -1, -1
        elif a[1] == b[0]:
            return 1, 1
        elif a[1] == b[1]:
            return 1, -1
        else:
            raise ValueError('无法确定方向,边不相连！')

    if vertices is None:
        ccw_direction = 1
    else:
        ccw_check = is_ccw(vertices[np.append(vertex_path, vertex_path[0])])
        ccw_direction = (ccw_check * 2) - 1

    # 填充实体列表
    vertex_path = np.asanyarray(vertex_path)
    entity_path = deque()
    for i in np.arange(len(vertex_path) + 1):
        vertex_path_pos = np.mod(np.arange(2) + i, len(vertex_path))
        vertex_index = vertex_path[vertex_path_pos]
        entity_index = graph.get_edge_data(*vertex_index)['entity_index']
        entity_path.append(entity_index)

    # 删除重复的实体
    entity_path = unique_ordered(entity_path)[::ccw_direction]

    # 遍历实体路径并反向实体以对齐此路径顺序
    round_trip = np.append(entity_path, entity_path[0])
    round_trip = zip(round_trip[:-1], round_trip[1:])
    for a, b in round_trip:
        da, db = edge_direction(entities[a].end_points, entities[b].end_points)
        entities[a].points = entities[a].points[::da]
        entities[b].points = entities[b].points[::db]
    return entity_path


def connected_open(graph):
    """
    找出图中连接的开放节点

    :param graph: 顶点连接的图
    :return: 两个集合,分别表示断开的节点和正常的节点
    """
    broken = set()
    for node, degree in graph.degree().items():
        if degree == 2:
            continue
        if node in broken:
            continue
        [broken.add(i) for i in nx.node_connected_component(graph, node)]
    okay = set(graph.nodes()).difference(broken)
    return broken, okay


def closed_paths(entities, vertices):
    '''
    生成闭合路径.路径是实体索引的列表

    首先使用图循环算法生成顶点路径,然后通过多个循环和条件将其转换为实体路径.
    这也会更改实体点的顺序,以便可以遍历路径而无需反转实体

    :param entities: 实体对象的列表
    :param vertices: 顶点的列表
    :return: 闭合路径的数组
    '''
    graph, closed = vertex_graph(entities)
    paths = deque(np.reshape(closed, (-1, 1)))
    vertex_paths = np.array(nx.cycles.cycle_basis(graph))

    for vertex_path in vertex_paths:
        if len(vertex_path) < 2:
            continue
        entity_path = vertex_to_entity_path(vertex_path,
                                            graph,
                                            entities,
                                            vertices)
        paths.append(np.array(entity_path))
    paths = np.array(paths)
    return paths


def arctan2_points(points):
    """
    计算点的角度,确保角度为正

    :param points: 点的数组
    :return: 角度的数组
    """
    angle = np.arctan2(*points.T[::-1])
    test = angle < 0.0
    angle[test] = (np.pi * 2) + angle[test]
    return angle


def discretize_path(entities, vertices, path, scale=1.0):
    '''
    返回顶点的列表,采样弧线/曲线为线段

    :param entities: 实体对象的列表
    :param vertices: 顶点的列表
    :param path: 路径的实体索引列表
    :param scale: 缩放比例
    :return: 离散化的顶点数组
    '''
    path_len = len(path)
    if path_len == 0:
        raise NameError('Cannot 离散 empty path!')
    if path_len == 1:
        return np.array(entities[path[0]].discrete(vertices))
    discrete = deque()
    for i, entity_id in enumerate(path):
        last = (i == (path_len - 1))
        current = entities[entity_id].discrete(vertices, scale=scale)
        slice = (int(last) * len(current)) + (int(not last) * -1)
        discrete.extend(current[:slice])
    discrete = np.array(discrete)

    return discrete


class PathSample:
    def __init__(self, points):
        # 确保输入数组是 numpy 数组
        self._points = np.array(points)
        # 找到每个线段的方向
        self._vectors = np.diff(self._points, axis=0)
        # 找到每个线段的长度
        self._norms = np.linalg.norm(self._vectors, axis=1)
        # 每个线段的单位向量
        nonzero = self._norms > tol.zero
        self._unit_vec = self._vectors.copy()
        self._unit_vec[nonzero] /= self._norms[nonzero].reshape((-1, 1))
        # 路径的总距离
        self.length = self._norms.sum()
        # 线段长度的累积和 注意这是排序的
        self._cum_norm = np.cumsum(self._norms)

    def sample(self, distances):
        # 返回cum_norm中每个样本需要插入的索引,以维护sorted属性
        positions = np.searchsorted(self._cum_norm, distances)
        positions = np.clip(positions, 0, len(self._unit_vec) - 1)
        offsets = np.append(0, self._cum_norm)[positions]
        # 需要从参考顶点移动的距离
        projection = distances - offsets
        # 找出需要投影的方向
        direction = self._unit_vec[positions]
        # 找出偏移的起始顶点
        origin = self._points[positions]
        # 线的参数方程
        resampled = origin + (direction * projection.reshape((-1, 1)))
        return resampled

    def truncate(self, distance):
        '''
        返回路径的截断版本,仅在终点处添加一个顶点
        '''
        position = np.searchsorted(self._cum_norm, distance)
        offset = distance - self._cum_norm[position - 1]
        if offset < tol.merge:
            truncated = self._points[:position + 1]
        else:
            vector = unitize(np.diff(self._points[np.arange(2) + position], axis=0).reshape(-1))
            vector *= offset
            endpoint = self._points[position] + vector
            truncated = np.vstack((self._points[:position + 1], endpoint))
        assert (np.linalg.norm(np.diff(truncated, axis=0), axis=1).sum() - distance) < tol.merge
        return truncated


def resample_path(points, count=None, step=None, step_round=True):
    '''
    给定一条路径上的 (n,d) 个点,重新采样它们,使得路径上每个重新采样点之间的距离恒定
    注意,这可能会在拐角处产生裁剪,因为原始顶点不保证在新的重新采样路径中

    只能指定 count 或 step 之一
    通过指定 count,可以生成均匀分布的结果 (np.linspace)
    通过指定 step,可以生成具有特定距离的结果 (np.arange)

    :param points: (n,d) 空间中的点序列
    :param count: 采样点的数量 (即 np.linspace)
    :param step: 路径上每步的距离 (即 np.arange)
    :return: resampled: (j,d) 路径上的点集
    '''
    points = np.array(points, dtype=float)
    # 根据参数 count 或 step 生成沿路径的样本
    if (count is not None) and (step is not None):
        raise ValueError('只能指定 step 或 count 之一')
    if (count is None) and (step is None):
        raise ValueError('必须指定 step 或 count 之一')

    sampler = PathSample(points)
    if step is not None and step_round:
        count = int(np.ceil(sampler.length / step))
    if count is not None:
        samples = np.linspace(0, sampler.length, count)
    elif step is not None:
        samples = np.arange(0, sampler.length, step)

    resampled = sampler.sample(samples)

    check = np.linalg.norm(points[[0, -1]] - resampled[[0, -1]], axis=1)
    assert check[0] < tol.merge
    if count is not None:
        assert check[1] < tol.merge

    return resampled
