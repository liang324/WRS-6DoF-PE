# Entities.py:基本的几何基元
# 设计意图:只存储顶点索引的引用并传递顶点
#         数组返回给需要它的函数.
#         这将所有顶点保存在一个外部列表中


import numpy as np
from .arc import discretize_arc, arc_center
from .curve import discretize_bezier, discretize_bspline
from ..util import replace_references

_HASH_LENGTH = 5


class Entity(object):
    def __init__(self,
                 points,
                 closed=None):
        """
        初始化实体对象

        :param points: 实体的点列表
        :param closed: 可选参数,指示实体是否闭合
        """
        self.points = np.asanyarray(points)
        if closed is not None:
            self.closed = closed

    @property
    def _class_id(self):
        """
        返回一个整数,该整数对类类型是唯一的
        请注意,如果定义的类以现有类相同的字母开头,则此实现将失败
        由于此函数被频繁调用,因此在速度和健壮性之间进行了权衡,速度获胜
        """
        return ord(self.__class__.__name__[0])

    @property
    def hash(self):
        '''
        返回一个对实体唯一的字符串
        如果存在两个相同的实体,可以通过比较此函数返回的字符串来删除它们
        '''
        hash = np.zeros(_HASH_LENGTH, dtype=int)
        hash[-2:] = self._class_id, int(self.closed)
        points_count = np.min([3, len(self.points)])
        hash[0:points_count] = np.sort(self.points)[-points_count:]
        return hash

    def to_dict(self):
        '''
        返回包含实体所有信息的字典
        '''
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'closed': self.closed}

    def rereference(self, replacement):
        '''
        给定替换字典,改变点以反映字典
        例如,如果 replacement = {0:107},self.points = [0,1902] 将变为 [107, 1902]
        '''
        self.points = replace_references(self.points, replacement)

    @property
    def closed(self):
        '''
        如果第一个点与终点相同,则实体是闭合的
        '''
        closed = (len(self.points) > 2 and np.equal(self.points[0], self.points[-1]))
        return closed

    @property
    def nodes(self):
        '''
        返回一个 (n,2) 的节点列表,或路径上的顶点
        请注意,此通用类函数假设所有参考点都在路径上,这对于线和三点弧是正确的
        如果要定义另一个类,该类不符合这种情况(例如,贝塞尔曲线的控制点)则需要实现特定于实体的版本

        拥有节点列表的目的是为了可以将它们作为边添加到图中,以便我们可以使用函数检查连通性、提取路径等.

        此函数的切片本质上只是平铺点,因此第一个和最后一个顶点不会重复.例如:

        self.points = [0,1,2]
        返回:      [[0,1], [1,2]]
        '''
        return np.column_stack((self.points, self.points)).reshape(-1)[1:-1].reshape((-1, 2))

    @property
    def end_points(self):
        '''
        返回第一个和最后一个点
        请注意,如果您定义了一个新的实体类,其中 self.points 中的第一个和最后一个顶点不是曲线的端点
        则需要为您的类实现此函数

        例如:
        self.points = [0,1,2]
        返回:      [0,2]
        '''
        return self.points[[0, -1]]

    @property
    def is_valid(self):
        """
        检查实体是否有效
        :return: 如果实体有效,返回 True
        """
        return True

    def reverse(self, direction=-1):
        '''
        反转当前实体
        '''
        self.points = self.points[::direction]


class Line(Entity):
    '''
    线或折线实体
    '''
    def discrete(self, vertices, scale=1.0):
        """
        获取离散化的顶点

        :param vertices: 顶点数组
        :param scale: 缩放比例
        :return: 离散化后的顶点
        """
        return vertices[self.points]

    @property
    def is_valid(self):
        """
        检查线实体是否有效

        :return: 如果线实体有效,返回 True
        """
        valid = np.any((self.points - self.points[0]) != 0)
        return valid


class Arc(Entity):
    @property
    def closed(self):
        """
        检查弧是否闭合

        :return: 如果弧闭合,返回 True
        """
        if hasattr(self, '_closed'):
            return self._closed
        return False

    @closed.setter
    def closed(self, value):
        """
        设置弧的闭合状态

        :param value: 布尔值,指示弧是否闭合
        """
        self._closed = bool(value)

    def discrete(self, vertices, scale=1.0):
        """
        获取离散化的弧

        :param vertices: 顶点数组
        :param scale: 缩放比例
        :return: 离散化后的弧
        """
        return discretize_arc(vertices[self.points],
                              close=self.closed,
                              scale=scale)

    def center(self, vertices):
        """
        计算弧的中心

        :param vertices: 顶点数组
        :return: 弧的中心点
        """
        return arc_center(vertices[self.points])


class Curve(Entity):
    @property
    def _class_id(self):
        #  返回一个唯一的整数,表示类的标识符
        return sum([ord(i) for i in self.__class__.__name__])

    @property
    def nodes(self):
        """
        返回曲线的节点

        :return: 节点列表
        """
        return [[self.points[0],
                 self.points[1]],
                [self.points[1],
                 self.points[-1]]]


class Bezier(Curve):
    def discrete(self, vertices, scale=1.0):
        """
        获取离散化的贝塞尔曲线

        :param vertices: 顶点数组
        :param scale: 缩放比例
        :return: 离散化后的贝塞尔曲线
        """
        return discretize_bezier(vertices[self.points], scale=scale)


class BSpline(Curve):
    def __init__(self, points, knots, closed=None):
        """
        初始化 B样条曲线对象

        :param points: 控制点列表
        :param knots: 节点向量
        :param closed: 是否闭合
        """
        self.points = points
        self.knots = knots

    def discrete(self, vertices, count=None, scale=1.0):
        """
        获取离散化的 B样条曲线

        :param vertices: 顶点数组
        :param count: 离散化点的数量
        :param scale: 缩放比例
        :return: 离散化后的 B样条曲线
        """
        result = discretize_bspline(control=vertices[self.points],
                                    knots=self.knots,
                                    count=count,
                                    scale=scale)
        return result

    def to_dict(self):
        """
        返回一个包含实体所有信息的字典
        :return: 实体信息字典
        """
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'knots': self.knots.tolist(),
                'closed': self.closed}
