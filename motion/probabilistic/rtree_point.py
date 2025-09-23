"""

这个文件的主要算法是基于 R-tree 数据结构实现的空间索引和最近邻查询.具体来说,它提供了以下两个主要功能
插入点 (insert 方法):将一个点(在高维空间中的坐标)插入到 R-tree 中,并且每个点都会有一个唯一的 id.这个 id 用于标识起点或目标点
最近邻查询 (nearest 方法):给定一个查询点,返回 R-tree 中距离该点最近的一个点的 id
总结:该文件实现了 R-tree 空间索引的基本操作,主要用于存储和查询空间中的点

"""

import numpy as np
from rtree import index


class RtreePoint():

    def __init__(self, dimension):
        p = index.Property()
        p.dimension = dimension
        p.storage = index.RT_Memory
        self._idx_rtp = index.Index(properties=p)
        self._dimension = dimension

    def insert(self, id, point):
        """
        The dimension of a point must be equal to dimension of the tree
        :param id
        :param point: a 1xn array
        :return:
        author: weiwei
        date: 20180520
        """
        if id == 'start':
            id = -1
        if id == 'goal':
            id = -2
        self._idx_rtp.insert(id, np.hstack((point, point)), point)

    def nearest(self, point):
        """
        The dimension of a point must be equal to dimension of the tree
        :param point: a 1xn list
        :return: id of the neareast point (use 'raw' for array; use True for objects
        author: weiwei
        date: 20180520
        """
        return_id = list(self._idx_rtp.nearest(np.hstack((point, point)), 1, objects=False))[0]
        if return_id == -1:
            return_id = 'start'
        if return_id == -2:
            return_id = 'goal'
        return return_id
