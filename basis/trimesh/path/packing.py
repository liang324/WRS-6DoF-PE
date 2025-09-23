import numpy as np
from collections import deque
from ..constants import log, time_function
from ..constants import tol_path as tol
from .polygons import polygons_obb, transform_polygon


class RectangleBin:
    '''
    2D BSP 树节点
    参考:http://www.blackpawn.com/texts/lightmaps/
    '''

    def __init__(self, bounds=None, size=None):
        """
        初始化 RectangleBin 对象

        :param bounds: 边界,格式为 (minx, miny, maxx, maxy)
        :param size: 尺寸,如果提供,将用于设置边界
        """
        self.child = [None] * 2
        # bounds: (minx, miny, maxx, maxy)
        self.bounds = bounds
        self.occupied = False
        if size != None:
            self.bounds = np.append([0, 0], size)

    def insert(self, rectangle_size):
        """
        插入一个矩形到当前的 BSP 节点中

        :param rectangle_size: 矩形的尺寸
        :return: 插入位置的坐标或 None
        """
        for child in self.child:
            if child != None:
                attempt = child.insert(rectangle_size)
                if attempt: return attempt
        if self.occupied:
            return None

        # 比较 bin 的大小和插入候选的大小
        size_test = bounds_to_size(self.bounds) - rectangle_size

        # 如果插入的矩形太大,返回 None
        if np.any(size_test < -tol.zero):
            return None

        # 如果矩形正好适合当前节点
        self.occupied = True
        if np.all(size_test < tol.zero):
            return self.bounds[0:2]

        # 如果矩形适合但空余空间太大,需要创建子节点
        vertical = size_test[0] > size_test[1]
        length = rectangle_size[not vertical]
        child_bounds = self.split(length, vertical)

        self.child[0] = RectangleBin(bounds=child_bounds[0])
        self.child[1] = RectangleBin(bounds=child_bounds[1])

        return self.child[0].insert(rectangle_size)

    def split(self, length=None, vertical=True):
        '''
        返回两个边界框,表示当前边界分割成两个较小的框

        :param length: 分割的长度
        :param vertical: 是否垂直分割
        :return: 两个新的边界框
        '''
        [left, bottom, right, top] = self.bounds
        if vertical:
            box = [[left, bottom, left + length, top],
                   [left + length, bottom, right, top]]
        else:
            box = [[left, bottom, right, bottom + length],
                   [left, bottom + length, right, top]]
        return box


def bounds_to_size(bounds):
    """
    将边界转换为尺寸

    :param bounds: 边界,格式为 (minx, miny, maxx, maxy)
    :return: 尺寸
    """
    return np.diff(np.reshape(bounds, (2, 2)), axis=0)[0]


def pack_rectangles(rectangles, sheet_size, shuffle=False):
    '''
    将小矩形打包到一个更大的矩形上,使用二叉空间分割树

    :param rectangles: (n,2) 数组,表示要打包的小矩形的 (宽度, 高度) 对
    :param sheet_size: (2) 数组,表示大矩形的 (宽度, 高度)
    :param shuffle: 布尔值,是否打乱小矩形的插入顺序,因为最终的打包密度取决于小矩形插入到大矩形上的顺序
    :return: 打包密度,偏移量数组,插入标记数组,消耗的矩形大小
    '''
    offset = np.zeros((len(rectangles), 2))
    inserted = np.zeros(len(rectangles), dtype=bool)
    box_order = np.argsort(np.sum(rectangles ** 2, axis=1))[::-1]
    area = 0.0
    density = 0.0
    if shuffle:
        shuffle_len = int(np.random.random() * len(rectangles)) - 1
        box_order[0:shuffle_len] = np.random.permutation(box_order[0:shuffle_len])

    sheet = RectangleBin(size=sheet_size)
    for index in box_order:
        insert_location = sheet.insert(rectangles[index])
        if insert_location != None:
            area += np.prod(rectangles[index])
            offset[index] += insert_location
            inserted[index] = True

    consumed_box = np.max((offset + rectangles)[inserted], axis=0)
    density = area / np.product(consumed_box)

    return density, offset[inserted], inserted, consumed_box


def pack_paths(paths, show=False):
    '''
    打包路径

    :param paths: 路径列表
    :param show: 布尔值,是否显示打包结果
    :return: 打包后的路径列表
    '''
    paths_full = deque()
    for path in paths:
        if 'quantity' in path.metadata:
            paths_full.extend([path.copy() for i in range(path.metadata['quantity'])])
        else:
            paths_full.append(path.copy())

    polygons = [i.polygons_closed[i.workdir[0]] for i in paths_full]
    inserted, transforms = multipack(np.array(polygons))
    for path, transform in zip(paths_full, transforms):
        path.transform(transform)
        if show: path.plot_discrete(show=False)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return paths_full


def multipack(polygons,
              sheet_size=None,
              iterations=50,
              density_escape=.985,
              buffer_dist=0.09,
              plot=False,
              return_all=False):
    '''
    通过随机排列插入顺序运行多次矩形打包

    :param polygons: 多边形列表
    :param sheet_size: (2) 数组,表示大矩形的 (宽度, 高度)
    :param iterations: 迭代次数
    :param density_escape: 打包密度达到此值时提前退出
    :param buffer_dist: 缓冲距离
    :param plot: 布尔值,是否绘制打包结果
    :param return_all: 布尔值,是否返回所有结果
    :return: 插入标记数组,变换矩阵
    '''
    transforms_obb, rectangles = polygons_obb(polygons)
    rectangles += 2.0 * buffer_dist
    polygon_area = np.array([p.area for p in polygons])
    tic = time_function()
    overall_density = 0
    if sheet_size == None:
        max_dim = np.max(rectangles, axis=0)
        sum_dim = np.sum(rectangles, axis=0)
        sheet_size = [sum_dim[0], max_dim[1] * 2]

    log.info('Packing %d polygons', len(polygons))
    for i in range(iterations):
        density, offset, inserted, sheet = pack_rectangles(rectangles,
                                                           sheet_size=sheet_size,
                                                           shuffle=(i != 0))

        if density > overall_density:
            overall_density = density
            overall_offset = offset
            overall_inserted = inserted
            overall_sheet = sheet
            if density > density_escape:
                break

    toc = time_function()
    log.info('Packing finished %i iterations in %f seconds', i + 1, toc - tic)
    log.info('%i/%i parts were packed successfully', np.sum(overall_inserted), len(polygons))
    log.info('Final rectangular density is %f.', overall_density)

    polygon_density = np.sum(polygon_area[overall_inserted]) / np.product(overall_sheet)
    log.info('Final polygonal density is %f.', polygon_density)

    transforms_obb = transforms_obb[overall_inserted]
    transforms_packed = transforms_obb.copy()
    transforms_packed.reshape(-1, 9)[:, [2, 5]] += overall_offset + buffer_dist

    if plot:
        transform_polygon(np.array(polygons)[overall_inserted],
                          transforms_packed,
                          plot=True)
    rectangles -= 2.0 * buffer_dist

    if return_all:
        return (overall_inserted,
                transforms_packed,
                transforms_obb,
                overall_sheet,
                rectangles[overall_inserted])

    return overall_inserted, transforms_packed


class Packer:
    def __init__(self, sheet_size=None):
        """
        初始化 Packer 对象

        :param sheet_size: (2) 数组,表示大矩形的 (宽度, 高度)
        """
        pass

    def add(self):
        # 添加一个新的矩形到打包器中
        pass
