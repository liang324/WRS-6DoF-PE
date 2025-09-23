import numpy as np
from . import util
from . import points
from . import creation
from .base import Trimesh
from .constants import log
from .triangles import windings_aligned


class Primitive(Trimesh):
    '''
    几何原始体是 Trimesh 的子类
    当请求顶点或面时,网格会被延迟生成
    '''

    def __init__(self, *args, **kwargs):
        '''
        初始化 Primitive 类的实例

        :param args: 位置参数列表,用于初始化 Trimesh
        :param kwargs: 关键字参数字典,用于初始化 Trimesh
        '''
        super(Primitive, self).__init__(*args, **kwargs)
        self._data.clear()
        self._validate = False

    @property
    def faces(self):
        '''
        获取原始体的面

        :return: 面的数组,形状为 (-1, 3)
        '''
        stored = self._cache['faces']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        # self._validate_face_normals()
        return self._cache['faces']

    @faces.setter
    def faces(self, values):
        '''
        设置原始体的面

        :param values: 面的值(不可变)
        '''
        log.warning('原始体的面是不可变的！不设置！')

    @property
    def vertices(self):
        '''
        获取原始体的顶点

        :return: 顶点的数组,形状为 (-1, 3)
        '''
        stored = self._cache['vertices']
        if util.is_shape(stored, (-1, 3)):
            return stored

        self._create_mesh()
        return self._cache['vertices']

    @vertices.setter
    def vertices(self, values):
        '''
        设置原始体的顶点

        :param values: 顶点的值(不可变)
        '''
        if values is not None:
            log.warning('原始体的顶点是不可变的！不设置！')

    @property
    def face_normals(self):
        '''
        获取原始体的面法线

        :return: 面法线的数组,形状为 (-1, 3)
        '''
        stored = self._cache['face_normals']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache['face_normals']

    @face_normals.setter
    def face_normals(self, values):
        '''
        设置原始体的面法线

        :param values: 面法线的值(不可变)
        '''
        if values is not None:
            log.warning('原始体的面法线是不可变的！不设置！')

    def _create_mesh(self):
        '''
        创建网格的方法

        :raises ValueError: 如果没有定义网格创建
        '''
        raise ValueError('原始体没有定义网格创建！')


class Sphere(Primitive):
    def __init__(self, *args, **kwargs):
        '''
        创建一个球体原始体,它是 Trimesh 的子类

        :param sphere_radius: float, 球体的半径
        :param sphere_center: (3,) float, 球体的中心
        :param subdivisions: int, 用于生成二十面体球的细分次数.默认值为 3
        '''
        super(Sphere, self).__init__(*args, **kwargs)
        if 'sphere_radius' in kwargs:
            self.sphere_radius = kwargs['sphere_radius']
        if 'sphere_center' in kwargs:
            self.sphere_center = kwargs['sphere_center']
        if 'subdivisions' in kwargs:
            self._data['subdivisions'] = int(kwargs['subdivisions'])
        else:
            self._data['subdivisions'] = 3
        self._unit_sphere = creation.icosphere(subdivisions=self._data['subdivisions'][0])

    @property
    def sphere_center(self):
        '''
        获取球体的中心

        :return: 球体中心的坐标数组,形状为 (3,)
        '''
        stored = self._data['center']
        if stored is None:
            return np.zeros(3)
        return stored

    @sphere_center.setter
    def sphere_center(self, values):
        '''
        设置球体的中心

        :param values: 球体中心的坐标数组
        '''
        self._data['center'] = np.asanyarray(values, dtype=np.float64)

    @property
    def sphere_radius(self):
        '''
        获取球体的半径

        :return: 球体的半径,默认为 1.0
        '''
        stored = self._data['radius']
        if stored is None:
            return 1.0
        return stored

    @sphere_radius.setter
    def sphere_radius(self, value):
        '''
        设置球体的半径

        :param value: 球体的半径
        '''
        self._data['radius'] = float(value)

    def _create_mesh(self):
        '''
        创建球体的网格

        :return: 更新缓存中的顶点、面和面法线
        '''
        ico = self._unit_sphere
        self._cache['vertices'] = ((ico.vertices * self.sphere_radius) + self.sphere_center)
        self._cache['faces'] = ico.faces
        self._cache['face_normals'] = ico.face_normals


class Box(Primitive):
    def __init__(self, *args, **kwargs):
        '''
        创建一个盒子原始体,它是 Trimesh 的子类

        :param box_extents: (3,) float, 盒子的尺寸
        :param box_transform: (4,4) float, 盒子的变换矩阵
        :param box_center: (3,) float, 便捷函数,用于更新盒子变换矩阵,仅包含平移矩阵
        '''
        super(Box, self).__init__(*args, **kwargs)
        if 'box_extents' in kwargs:
            self.box_extents = kwargs['box_extents']
        if 'box_transform' in kwargs:
            self.box_transform = kwargs['box_transform']
        if 'box_center' in kwargs:
            self.box_center = kwargs['box_center']
        self._unit_box = creation.box()

    @property
    def box_center(self):
        '''
        获取盒子的中心

        :return: 盒子中心的坐标数组,形状为 (3,)
        '''
        return self.box_transform[0:3, 3]

    @box_center.setter
    def box_center(self, values):
        '''
        设置盒子的中心

        :param values: 盒子中心的坐标数组
        '''
        transform = self.box_transform
        transform[0:3, 3] = values
        self._data['box_transform'] = transform

    @property
    def box_extents(self):
        '''
        获取盒子的尺寸

        :return: 盒子的尺寸数组,形状为 (3,)
        '''
        stored = self._data['box_extents']
        if util.is_shape(stored, (3,)):
            return stored
        return np.ones(3)

    @box_extents.setter
    def box_extents(self, values):
        '''
        设置盒子的尺寸

        :param values: 盒子的尺寸数组
        '''
        self._data['box_extents'] = np.asanyarray(values, dtype=np.float64)

    @property
    def box_transform(self):
        stored = self._data['box_transform']
        if util.is_shape(stored, (4, 4)):
            return stored
        return np.eye(4)

    @box_transform.setter
    def box_transform(self, matrix):
        '''
        获取盒子的变换矩阵

        :return: 盒子的变换矩阵,形状为 (4, 4)
        '''
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['box_transform'] = matrix

    @property
    def is_oriented(self):
        '''
        检查盒子是否经过旋转

        :return: 如果盒子经过旋转,返回 True；否则返回 False
        '''
        if util.is_shape(self.box_transform, (4, 4)):
            return not np.allclose(self.box_transform[0:3, 0:3], np.eye(3))
        else:
            return False

    def _create_mesh(self):
        '''
        创建盒子的网格

        :return: 更新缓存中的顶点、面和面法线
        '''
        log.debug('为盒子原始体创建网格')
        box = self._unit_box
        vertices, faces, normals = box.vertices, box.faces, box.face_normals
        vertices = points.transform_points(vertices * self.box_extents, self.box_transform)
        normals = np.dot(self.box_transform[0:3, 0:3], normals.T).T
        aligned = windings_aligned(vertices[faces[:1]], normals[:1])[0]
        if not aligned:
            faces = np.fliplr(faces)
        # 对于原始体,顶点和面是从其他信息派生的
        # 因此它们存储在缓存中,而不是数据存储中
        self._cache['vertices'] = vertices
        self._cache['faces'] = faces
        self._cache['face_normals'] = normals


class Cylinder(Primitive):
    def __init__(self, *args, **kwargs):
        """
        创建一个圆柱体原始体,它是 Trimesh 的子类

        :param radius: float, 圆柱体的半径
        :param height: float, 圆柱体的高度
        :param sections: int, 圆周上的分段数
        :param homomat: 可选的齐次变换矩阵
        """
        super(Cylinder, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'sections' in kwargs:
            self.sections = kwargs['sections']
        else:
            self.sections = 12
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        计算圆柱体原始体的解析体积

        :return: float, 圆柱体的体积
        """
        volume = ((np.pi * self.radius ** 2) * self.height)
        return volume

    def buffer(self, distance):
        """
        返回一个覆盖源圆柱体的圆柱体原始体,距离为 distance:
        半径增加 distance,高度增加两倍的 distance

        :param distance: float, 用于膨胀圆柱体半径和高度的距离
        :return: Cylinder, 被 distance 膨胀的圆柱体原始体
        """
        distance = float(distance)
        buffered = Cylinder(height=self.height + distance * 2, radius=self.radius + distance, sections=self.sections,
                            homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        """
        创建圆柱体的网格

        :return: 更新缓存中的顶点、面和面法线
        """
        log.debug('创建圆柱体网格,半径=%f,高度=%f,分段数=%d', self.radius, self.height, self.sections)
        mesh = creation.cylinder(radius=self.radius, height=self.height, sections=self.sections, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Extrusion(Primitive):
    def __init__(self, *args, **kwargs):
        '''
        创建一个拉伸原始体,它是 Trimesh 的子类

        :param extrude_polygon: shapely.geometry.Polygon, 要拉伸的多边形
        :param extrude_transform: (4,4) float, 拉伸后要应用的变换矩阵
        :param extrude_height: float, 拉伸多边形的高度
        '''
        super(Extrusion, self).__init__(*args, **kwargs)
        if 'extrude_polygon' in kwargs:
            self.extrude_polygon = kwargs['extrude_polygon']
        if 'extrude_transform' in kwargs:
            self.extrude_transform = kwargs['extrude_transform']
        if 'extrude_height' in kwargs:
            self.extrude_height = kwargs['extrude_height']

    @property
    def extrude_transform(self):
        '''
        获取拉伸后的变换矩阵

        :return: (4,4) 变换矩阵
        '''
        stored = self._data['extrude_transform']
        if np.shape(stored) == (4, 4):
            return stored
        return np.eye(4)

    @extrude_transform.setter
    def extrude_transform(self, matrix):
        '''
        设置拉伸后的变换矩阵

        :param matrix: (4,4) 变换矩阵
        :raises ValueError: 如果矩阵的形状不是 (4, 4)
        '''
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['extrude_transform'] = matrix

    @property
    def extrude_height(self):
        '''
        获取拉伸的高度

        :return: float, 拉伸的高度
        :raises ValueError: 如果未指定拉伸高度
        '''
        stored = self._data['extrude_height']
        if stored is None:
            raise ValueError('extrude height not specified!')
        return stored.copy()[0]

    @extrude_height.setter
    def extrude_height(self, value):
        '''
        设置拉伸的高度

        :param value: float, 拉伸的高度
        '''
        self._data['extrude_height'] = float(value)

    @property
    def extrude_polygon(self):
        '''
        获取要拉伸的多边形

        :return: shapely.geometry.Polygon, 要拉伸的多边形
        :raises ValueError: 如果未指定拉伸多边形
        '''
        stored = self._data['extrude_polygon']
        if stored is None:
            raise ValueError('extrude polygon not specified!')
        return stored[0]

    @extrude_polygon.setter
    def extrude_polygon(self, value):
        '''
        设置要拉伸的多边形

        :param value: shapely.geometry.Polygon, 要拉伸的多边形
        '''
        polygon = creation.validate_polygon(value)
        self._data['extrude_polygon'] = polygon

    @property
    def extrude_direction(self):
        '''
        获取拉伸的方向

        :return: numpy 数组,表示拉伸方向的向量
        '''
        direction = np.dot(self.extrude_transform[:3, :3], [0.0, 0.0, 1.0])
        return direction

    def slide(self, distance):
        '''
        沿拉伸方向滑动一定距离

        :param distance: float, 滑动的距离
        '''
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.extrude_transform.copy(), translation.copy())
        self.extrude_transform = new_transform

    def _create_mesh(self):
        '''
        创建拉伸原始体的网格

        :return: 更新缓存中的顶点、面和面法线
        '''
        log.debug('为拉伸原始体创建网格')
        mesh = creation.extrude_polygon(self.extrude_polygon, self.extrude_height)
        mesh.apply_transform(self.extrude_transform)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Cone(Primitive):

    def __init__(self, *args, **kwargs):
        """
        创建一个圆锥原始体,它是 Trimesh 的子类

        :param radius: float, 圆锥的半径
        :param height: float, 圆锥的高度
        :param sections: int, 圆周上的分段数

        author: weiwei
        date: 20191228
        """
        super(Cone, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'sections' in kwargs:
            self.sections = kwargs['sections']
        else:
            self.sections = 12
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        计算圆锥原始体的解析体积

        :return: float, 圆锥的体积

        author: weiwei
        date: 20191228osaka
        """
        volume = ((np.pi * self.radius ** 2) * self.height) / 3
        return volume

    def buffer(self, distance):
        """
        返回一个覆盖源圆锥的圆锥原始体,距离为 distance
        半径增加 distance,高度增加两倍的 distance

        :param distance: float, 用于膨胀圆锥半径和高度的距离
        :return: Cone, 被 distance 膨胀的圆锥原始体

        author: weiwei
        date: 20191228osaka
        """
        distance = float(distance)
        buffered = Cone(height=self.height + distance * 2, radius=self.radius + distance, sections=self.sections,
                        homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        """
        创建圆锥的网格

        :return: 更新缓存中的顶点、面和面法线
        """
        log.debug('创建圆锥网格,半径=%f,高度=%f,分段数=%d', self.radius, self.height, self.sections)
        mesh = creation.cone(radius=self.radius, height=self.height, sections=self.sections, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Capsule(Primitive):
    def __init__(self, *args, **kwargs):
        """
        创建一个胶囊原始体,它是 Trimesh 的子类

        :param radius: float, 胶囊的半径
        :param height: float, 胶囊的高度
        :param count: [int, int], 圆柱和半球的分段数
        :param homomat: 4x4 变换矩阵

        author: weiwei
        date: 20191228
        """
        super(Capsule, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'count' in kwargs:
            self.count = kwargs['count']
        else:
            self.count = [8, 8]
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        计算胶囊原始体的解析体积

        :return: float, 胶囊的体积

        author: weiwei
        date: 20191228osaka
        """
        volume = (np.pi * self.radius ** 3) * 3 / 4 + (np.pi * self.radius ** 2) * self.height
        return volume

    def buffer(self, distance):
        """
        返回一个覆盖源胶囊的胶囊原始体,距离为 distance

        :param distance: float, 用于膨胀胶囊半径和高度的距离
        :return: Capsule, 被 distance 膨胀的胶囊原始体

        author: weiwei
        date: 20191228osaka
        """
        distance = float(distance)
        buffered = Capsule(height=self.height + distance * 2, radius=self.radius + distance, count=self.count,
                           homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        """
        创建胶囊的网格

        :return: 更新缓存中的顶点、面和面法线
        """
        log.debug('创建胶囊网格,半径=%f,高度=%f,分段数=%d', self.radius, self.height, self.count)
        mesh = creation.capsule(radius=self.radius, height=self.height, count=self.count, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
