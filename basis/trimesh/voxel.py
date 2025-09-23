import numpy as np
from .grouping import unique_float
from .points import plot_points
from collections import deque


class Voxel:
    def __init__(self, mesh, pitch):
        """
        初始化 Voxel 类的实例

        :param mesh: 输入的网格对象
        :param pitch: 体素网格的间距
        """
        self._run = mesh_to_run(mesh, pitch)
        self._raw = None

    @property
    def raw(self):
        '''
        从内部的游程编码数据生成一个原始的3D布尔数组
        '''
        if self._raw is None:
            self._raw = run_to_raw(**self.run)
        return self._raw

    @property
    def run(self):
        """
        获取游程编码的体素网格数据

        :return: 包含体素网格信息的字典
        """
        return self._run

    @property
    def pitch(self):
        """
        获取体素网格的间距

        :return: 体素网格的间距值
        """
        return self.run['pitch']

    @property
    def origin(self):
        """
        获取体素网格的原点

        :return: 体素网格的原点坐标
        """
        return self.run['origin']

    def volume(self):
        """
        计算体素网格的体积

        :return: 体积值
        """
        volume = self.raw.sum() * (self.pitch ** 2)
        return volume

    def show(self):
        """
        显示体素网格
        """
        plot_raw(self.raw, **self.run)


def run_to_raw(shape, index_xy, index_z, **kwargs):
    """
    将游程编码数据转换为原始的3D布尔数组

    :param shape: 体素网格的形状
    :param index_xy: XY平面的索引
    :param index_z: Z轴的索引
    :return: 原始的3D布尔数组
    """
    raw = np.zeros(shape, dtype=bool)
    for xy, z in zip(index_xy, index_z):
        for z_start, z_end in np.reshape(z, (-1, 2)):
            raw[xy[0], xy[1]][z_start:z_end] = True
    return raw


def mesh_to_run(mesh, pitch):
    '''
    将网格转换为游程编码的体素网格

    这是通过射线测试实现的,返回交点
    这很容易转换为原始的3D布尔体素数组
    '''
    bounds = mesh.bounds / pitch
    bounds[0] = np.floor(bounds[0]) * pitch
    bounds[1] = np.ceil(bounds[1]) * pitch

    x_grid = np.arange(*bounds[:, 0], step=pitch)
    y_grid = np.arange(*bounds[:, 1], step=pitch)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))

    ray_origins = np.column_stack((grid, np.tile(bounds[0][2], len(grid))))
    ray_origins += [pitch * .5, pitch * .5, -pitch]
    ray_vectors = np.tile([0.0, 0.0, 1.0], (len(grid), 1))
    rays = np.column_stack((ray_origins, ray_vectors)).reshape((-1, 2, 3))

    hits = mesh.ray.intersects_location(rays)
    raw_shape = np.ptp(bounds / pitch, axis=0).astype(int)
    grid_origin = bounds[0]
    grid_index = ((grid / pitch) - (grid_origin[0:2] / pitch)).astype(int)

    run_z = deque()
    run_xy = deque()

    for i, hit in enumerate(hits):
        if len(hit) == 0:
            continue

        z = hit[:, 2] - grid_origin[2]
        # 如果射线正好击中边缘,命中结果中会有重复条目(针对两个三角形)
        z = unique_float(z)

        index_z = np.round(z / pitch).astype(int)
        # 如果命中点在边缘,会返回两次 np.unique 调用返回排序后的唯一索引
        index_z.sort()

        if np.mod(len(index_z), 2) != 0:
            # 这可能是命中顶点的情况
            index_z = index_z[[0, -1]]

        run_z.append(index_z)
        run_xy.append(grid_index[i])

    result = {'shape': raw_shape,
              'index_xy': np.array(run_xy),
              'index_z': np.array(run_z),
              'origin': grid_origin,
              'pitch': pitch}
    return result


def plot_raw(raw, pitch, origin, **kwargs):
    """
    绘制原始的3D布尔体素数组

    :param raw: 原始的3D布尔数组
    :param pitch: 体素网格的间距
    :param origin: 体素网格的原点
    """
    render = np.column_stack(np.nonzero(raw)) * pitch + origin
    plot_points(render)
