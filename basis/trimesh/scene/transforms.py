import numpy as np
import time
import networkx as nx
from ..transformations import homomat_from_quaternion, rotation_matrix


class TransformForest:
    def __init__(self, base_frame='world'):
        """
        初始化 TransformForest 类的实例

        :param base_frame: str,基础框架名称,默认为 'world'
        """
        self.transforms = EnforcedForest()
        self.base_frame = base_frame
        self._paths = {}

    def update(self,
               frame_to,
               frame_from=None,
               **kwargs):
        '''
        更新树中的变换

        :param frame_from: 可散列对象,通常是字符串(例如 'world')
                           如果为 None,则设置为 self.base_frame
        :param frame_to:   可散列对象,通常是字符串(例如 'mesh_0')

        其他参数(可以组合使用)
        ---------
        :param matrix:      (4,4) 数组
        :param quaternion:  (4) 四元数
        :param axis:        (3) 数组
        :param angle:       float,弧度
        :param translation: (3) 数组
        '''
        if frame_from is None:
            frame_from = self.base_frame
        matrix = kwargs_to_matrix(**kwargs)
        changed = self.transforms.add_edge(frame_from,
                                           frame_to,
                                           attr_dict={'matrix': matrix,
                                                      'time': time.time()})
        if changed:
            self._paths = {}

    def export(self):
        """
        导出当前变换树的边列表

        :return: export: 边列表,其中每个边的矩阵属性被转换为列表
        """
        export = nx.to_edgelist(self.transforms)
        for e in export:
            e[2]['matrix'] = np.array(e[2]['matrix']).tolist()
        return export

    def get(self,
            frame_to,
            frame_from=None):
        '''
        获取从一个框架到另一个框架的变换,假设它们在变换树中是连接的

        如果框架不连接,将引发 NetworkXNoPath 错误
        :param frame_from: 可散列对象,通常是字符串(例如 'world')
                           如果为 None,则设置为 self.base_frame
        :param frame_to:   可散列对象,通常是字符串(例如 'mesh_0')
        :return: transform: (4,4) 齐次变换矩阵
        '''

        if frame_from is None:
            frame_from = self.base_frame
        transform = np.eye(4)
        path = self._get_path(frame_from, frame_to)

        for i in range(len(path) - 1):
            data, direction = self.transforms.get_edge_data_direction(path[i],
                                                                      path[i + 1])
            matrix = data['matrix']
            if direction < 0:
                matrix = np.linalg.inv(matrix)
            transform = np.dot(transform, matrix)
        return transform

    def __getitem__(self, key):
        """
        获取指定键的变换矩阵

        :param key: 键,通常是一个字符串,表示帧的名称
        :return: 变换矩阵
        """
        return self.get(key)

    def __setitem__(self, key, value):
        """
        设置指定键的变换矩阵

        :param key: 键,通常是一个字符串,表示帧的名称
        :param value: 变换矩阵,必须是 (4, 4) 的数组
        """
        value = np.asanyarray(value)
        if value.shape != (4, 4):
            raise ValueError('矩阵必须是 (4, 4) 的数组！')
        return self.update(key, matrix=value)

    def clear(self):
        """
        清除所有变换和路径缓存
        """
        self.transforms = EnforcedForest()
        self._paths = {}

    def _get_path(self,
                  frame_from,
                  frame_to):
        '''
        查找两个帧之间的路径,可以从缓存路径或变换图中获取

        :param frame_from: 起始帧的键,通常是字符串,例如 'world'
        :param frame_to: 目标帧的键,通常是字符串,例如 'mesh_0'
        :return: 路径,帧键的列表,例如 ['mesh_finger', 'mesh_hand', 'world']
        '''
        key = (frame_from, frame_to)
        if not (key in self._paths):
            path = self.transforms.shortest_path_undirected(frame_from, frame_to)
            self._paths[key] = path
        return self._paths[key]


class EnforcedForest(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        """
        初始化 EnforcedForest 类的实例
        """
        self.flags = {'strict': False, 'assert_forest': False}

        for k, v in self.flags.items():
            if k in kwargs:
                self.flags[k] = bool(kwargs[k])
                kwargs.pop(k, None)

        super(self.__class__, self).__init__(*args, **kwargs)
        # 保持图的第二个并行但无向的副本
        # 所有用于翻转有向图的networkx方法
        # 转换成无向图非常慢,所以我们做了少量的簿记
        self._undirected = nx.Graph()

    def add_edge(self, u, v, *args, **kwargs):
        """
        添加一条边到图中

        :param u: 起始节点
        :param v: 目标节点
        :return: 如果图结构发生变化,返回 True,否则返回 False
        """
        changed = False
        if u == v:
            if self.flags['strict']:
                raise ValueError('边必须在两个不同的节点之间！')
            return changed
        if self._undirected.has_edge(u, v):
            self.remove_edges_from([[u, v], [v, u]])
        elif len(self.nodes()) > 0:
            try:
                path = nx.shortest_path(self._undirected, u, v)
                if self.flags['strict']:
                    raise ValueError('节点之间存在多条边路径！')
                self.disconnect_path(path)
                changed = True
            except (nx.NetworkXError, nx.NetworkXNoPath):
                pass
        self._undirected.add_edge(u, v)
        super(self.__class__, self).add_edge(u, v, *args, **kwargs)

        if self.flags['assert_forest']:
            # 这很慢,但可以非常确保结构正确,因此主要用于测试
            assert nx.is_forest(nx.Graph(self))
        return changed

    def add_edges_from(self, *args, **kwargs):
        """
        禁止使用 add_edges_from 方法,必须使用 add_edge 方法

        :raises ValueError: 总是抛出异常,提示使用 add_edge 方法
        """
        raise ValueError('EnforcedTree 需要使用 add_edge 方法！')

    def add_path(self, *args, **kwargs):
        """
        禁止使用 add_path 方法,必须使用 add_edge 方法

        :raises ValueError: 总是抛出异常,提示使用 add_edge 方法
        """
        raise ValueError('EnforcedTree 需要使用 add_edge 方法！')

    def remove_edge(self, *args, **kwargs):
        """
        从图中移除一条边,同时从无向图副本中移除

        :param args: 边的起始节点和目标节点
        :param kwargs: 其他参数
        """
        super(self.__class__, self).remove_edge(*args, **kwargs)
        self._undirected.remove_edge(*args, **kwargs)

    def remove_edges_from(self, *args, **kwargs):
        """
        从图中移除多条边,同时从无向图副本中移除

        :param args: 边的集合
        :param kwargs: 其他参数
        """
        super(self.__class__, self).remove_edges_from(*args, **kwargs)
        self._undirected.remove_edges_from(*args, **kwargs)

    def disconnect_path(self, path):
        """
        断开路径上的所有边

        :param path: 路径,帧键的列表
        """
        ebunch = np.array([[path[0], path[1]]])
        ebunch = np.vstack((ebunch, np.fliplr(ebunch)))
        self.remove_edges_from(ebunch)

    def shortest_path_undirected(self, u, v):
        """
        查找两个节点之间的最短路径(无向图)

        :param u: 起始节点
        :param v: 目标节点
        :return: 最短路径,节点列表
        """
        path = nx.shortest_path(self._undirected, u, v)
        return path

    def get_edge_data_direction(self, u, v):
        """
        获取边的数据和方向

        :param u: 起始节点
        :param v: 目标节点
        :return: 边的数据和方向(1 或 -1)
        :raises ValueError: 如果边不存在
        """
        if self.has_edge(u, v):
            direction = 1
        elif self.has_edge(v, u):
            direction = -1
        else:
            raise ValueError('Edge doesnt exist!')
        data = self.get_edge_data(*[u, v][::direction])
        return data, direction


def path_to_edges(path):
    '''
    将一个路径转换为边的集合

    :param path: (n) 路径,节点列表
    :return: (2(n-1)) 边的集合
    '''
    return np.column_stack((path, path)).reshape(-1)[1:-1].reshape((-1, 2))


def kwargs_to_matrix(**kwargs):
    '''
    将一组关键字参数转换为变换矩阵

    :param kwargs: 关键字参数,可以包括 'matrix', 'quaternion', 'axis', 'angle', 'translation'
    :return: 变换矩阵
    :raises ValueError: 如果无法更新变换
    '''
    matrix = np.eye(4)
    if 'matrix' in kwargs:
        # 矩阵优先于其他选项
        matrix = kwargs['matrix']
    elif 'quaternion' in kwargs:
        matrix = homomat_from_quaternion(kwargs['quaternion'])
    elif ('axis' in kwargs) and ('angle' in kwargs):
        matrix = rotation_matrix(kwargs['angle'],
                                 kwargs['axis'])
    else:
        raise ValueError('无法更新变换!')

    if 'translation' in kwargs:
        # translation 可以与任何指定变换的方法结合使用
        # 如果同时传递了矩阵和 translation,我们将 translation 加在一起而不是选择一个
        matrix[0:3, 3] += kwargs['translation']
    return matrix
