import numpy as np
from colorsys import hsv_to_rgb
from collections import deque
from .util import is_sequence, is_shape, Cache, DataStore
from .constants import log

COLORS = {'red': [205, 59, 34, 255],
          'purple': [150, 111, 214, 255],
          'blue': [119, 158, 203, 255],
          'brown': [160, 85, 45, 255]}

COLOR_DTYPE = np.dtype(np.uint8)
DEFAULT_COLOR = np.array(COLORS['purple'], dtype=COLOR_DTYPE)
RED_COLOR = np.array(COLORS['red'], dtype=COLOR_DTYPE)


class VisualAttributes(object):
    '''
    保存网格的视觉属性(通常是颜色)

    这个类需要重写,因为它的实现有些混乱
    '''

    def __init__(self, mesh=None, dtype=None, **kwargs):
        '''
        初始化 VisualAttributes 类

        :param mesh: Trimesh 对象,表示网格
        :param dtype: 数据类型,用于颜色数组
        :param kwargs: 其他参数,用于颜色设置
        '''
        self.mesh = mesh
        self._validate = True
        self._data = DataStore()
        self._cache = Cache(id_function=self._data.md5)
        if dtype is None:
            dtype = COLOR_DTYPE
        self.dtype = dtype
        colors = _kwargs_to_color(mesh, **kwargs)
        self.vertex_colors, self.face_colors = colors

    def choose(self):
        '''
        如果面和顶点颜色都定义了,选择其中之一
        '''
        if all(self._set.values()):
            sig_face = self._data['face_colors'].ptp(axis=0).sum()
            sig_vertex = self._cache['vertex_colors'].ptp(axis=0).sum()
            if sig_face > sig_vertex:
                self.vertex_colors = None
            else:
                self.face_colors = None

    @property
    def _set(self):
        """
        返回一个字典,指示面和顶点颜色是否已设置
        """
        result = {'face': is_shape(self._data['face_colors'], (-1, (3, 4))),
                  'vertex': is_shape(self._cache['vertex_colors'], (-1, (3, 4)))}
        return result

    @property
    def defined(self):
        """
        返回一个布尔值,指示视觉属性是否已定义
        """
        defined = np.any(self._set.values())
        defined = defined and self.mesh is not None
        return defined

    @property
    def transparency(self):
        '''
        返回视觉属性是否包含任何透明度

        :return: transparency: bool,视觉属性是否包含透明度
        '''
        cached = self._cache.get('transparency')
        if cached is not None:
            return cached
        transparency = False
        color_max = (2 ** (COLOR_DTYPE.itemsize * 8)) - 1
        if self._set['face']:
            transparency = (is_shape(self._data['face_colors'], (-1, 4)) and
                            np.any(self._data['face_colors'][:, 3] < color_max))
        elif self._set['vertex']:
            transparency = (is_shape(self._data['vertex_colors'], (-1, 4)) and
                            np.any(self._cache['vertex_colors'][:, 3] < color_max))
        return self._cache.set(key='transparency',
                               value=bool(transparency))

    def md5(self):
        '''
        返回数据存储的 MD5 哈希值
        '''
        return self._data.md5()

    @property
    def face_colors(self):
        '''
        获取面颜色

        :return: 面颜色数组
        '''
        stored = self._data['face_colors']
        if is_shape(stored, (len(self.mesh.faces), (3, 4))):
            return stored
        log.debug('Returning default colors for faces.')
        self._data['face_colors'] = np.tile(DEFAULT_COLOR, (len(self.mesh.faces), 1))
        return self._data['face_colors']

    @face_colors.setter
    def face_colors(self, values):
        '''
        设置面颜色

        :param values: 面颜色数组
        '''
        values = np.asanyarray(values)
        if values.shape in ((3,), (4,)):
            # 将单个RGB/RGBa颜色传递给setter的情况 我们将此颜色应用于所有面
            values = np.tile(values, (len(self.mesh.faces), 1))
        self._data['face_colors'] = rgba(values, dtype=self.dtype)

    @property
    def vertex_colors(self):
        '''
        获取顶点颜色

        :return: 顶点颜色数组
        '''
        cached = self._cache['vertex_colors']
        if is_shape(cached, (len(self.mesh.vertices), (3, 4))):
            return cached
        log.debug('从face颜色生成的顶点颜色')
        colors = face_to_vertex_color(self.mesh, self.face_colors)
        self._cache['vertex_colors'] = colors
        return colors

    @vertex_colors.setter
    def vertex_colors(self, values):
        '''
        设置顶点颜色

        :param values: 顶点颜色数组
        '''
        self._cache['vertex_colors'] = rgba(values, dtype=self.dtype)

    def update_faces(self, mask):
        '''
        更新面颜色

        :param mask: 布尔数组,指示哪些面需要更新
        '''
        stored = self._data['face_colors']
        if not is_shape(stored, (-1, (3, 4))):
            return
        try:
            self._data['face_colors'] = stored[mask]
        except:
            log.warning('Face colors not updated', exc_info=True)

    def update_vertices(self, mask):
        '''
        更新顶点颜色

        :param mask: 布尔数组,指示哪些顶点需要更新
        '''
        stored = self._data['vertex_colors']
        if not is_shape(stored, (-1, (3, 4))):
            return
        try:
            self._data['vertex_colors'] = stored[mask]
        except:
            log.debug('Vertex colors not updated', exc_info=True)

    def subsets(self, faces_sequence):
        '''
        创建视觉属性的子集

        :param faces_sequence: 面序列
        :return: 视觉属性子集数组
        '''
        result = [VisualAttributes() for i in range(len(faces_sequence))]
        if self._set['face']:
            face = self._data['face_colors']
            for i, f in enumerate(faces_sequence):
                result[i].face_colors = face[list(f)]
        return np.array(result)

    def union(self, others):
        '''
        合并多个视觉属性

        :param others: 其他视觉属性
        :return: 合并后的视觉属性
        '''
        return visuals_union(np.append(self, others))


def _kwargs_to_color(mesh, **kwargs):
    '''
    给定一组关键字参数,检查是否有任何参数名称引用颜色,并匹配网格的维度

    :param mesh: Trimesh 对象,表示网格
    :param kwargs: 关键字参数,可能包含颜色信息
    :return: 包含顶点颜色和面颜色的列表
    '''

    def pick_option(vf):
        '''
        从给定的选项中选择一个最佳选项

        :param vf: 选项列表
        :return: 最佳选项列表
        '''
        if any(i is None for i in vf):
            return vf
        result = [None, None]
        signal = [i.ptp(axis=0).sum() for i in vf]
        signal_max = np.argmax(signal)
        result[signal_max] = vf[signal_max]
        return result

    def pick_color(sequence):
        '''
        从颜色序列中选择一个最佳颜色

        :param sequence: 颜色序列
        :return: 最佳颜色
        '''
        if len(sequence) == 0:
            return None
        elif len(sequence) == 1:
            return sequence[0]
        else:
            signal = [i.ptp(axis=0).sum() for i in sequence]
            signal_max = np.argmax(signal)
            return sequence[signal_max]

    if mesh is None:
        result = [None, None]
        if 'face_colors' in kwargs:
            result[1] = np.asanyarray(kwargs['face_colors'])
        if 'vertex_colors' in kwargs:
            result[0] = np.asanyarray(kwargs['vertex_colors'])
        return result

    vertex = deque()
    face = deque()

    for key in kwargs.keys():
        if not ('color' in key):
            continue
        value = np.asanyarray(kwargs[key])
        if len(value) == len(mesh.vertices):
            vertex.append(value)
        elif len(value) == len(mesh.faces):
            face.append(value)
    return pick_option([pick_color(i) for i in [vertex, face]])


def visuals_union(visuals, *args):
    '''
    合并多个视觉属性对象

    :param visuals: 视觉属性对象列表
    :param args: 其他视觉属性对象
    :return: 合并后的 VisualAttributes 对象
    '''
    visuals = np.append(visuals, args)
    color = {'face_colors': None, 'vertex_colors': None}

    vertex_ok = True
    vertex = [None] * len(visuals)

    face_ok = True
    face = [None] * len(visuals)

    for i, v in enumerate(visuals):
        face_ok = face_ok and v._set['face']
        vertex_ok = vertex_ok and v._set['vertex']

        if face_ok:
            if v.mesh is None:
                # 如果mesh为None,不要强制a 检查颜色的尺寸
                face[i] = rgba(v._data['face_colors'])
            else:
                face[i] = rgba(v.face_colors)
        if vertex_ok:
            if v.mesh is None:
                vertex[i] = rgba(v._data['vertex_colors'])
            else:
                vertex[i] = rgba(v.vertex_colors)

    if face_ok:
        color['face_colors'] = np.vstack(face)
    if vertex_ok:
        color['vertex_colors'] = np.vstack(vertex)
    return VisualAttributes(**color)


def color_to_float(color, dtype=None):
    '''
    将颜色转换为浮点数表示

    :param color: 颜色数组
    :param dtype: 数据类型,默认为 None
    :return: 浮点数表示的颜色数组
    '''
    color = np.asanyarray(color)
    if dtype is None:
        dtype = color.dtype
    else:
        color = color.astype(dtype)
    if dtype.kind in 'ui':
        signed = int(dtype.kind == 'i')
        color_max = float((2 ** ((dtype.itemsize * 8) - signed)) - 1)
        color = color.astype(float) / color_max
    return color


def rgba(colors, dtype=None):
    '''
    将 RGB 颜色转换为 RGBA 颜色

    :param colors: (n,[3|4]) RGB 或 RGBA 颜色集合
    :param dtype: 数据类型,默认为 None
    :return: (n,4) RGBA 颜色集合
    '''
    if not is_sequence(colors):
        return
    if dtype is None:
        dtype = COLOR_DTYPE
    colors = np.asanyarray(colors, dtype=dtype)
    if is_shape(colors, (-1, 3)):
        opaque = (2 ** (np.dtype(dtype).itemsize * 8)) - 1
        colors = np.column_stack((colors, opaque * np.ones(len(colors)))).astype(dtype)
    return colors


def random_color(dtype=COLOR_DTYPE):
    '''
    使用指定的数据类型返回一个随机的 RGB 颜色

    :param dtype: 数据类型,默认为 COLOR_DTYPE
    :return: 随机生成的 RGB 颜色
    '''
    hue = np.random.random() + .61803
    hue %= 1.0
    color = np.array(hsv_to_rgb(hue, .99, .99))
    if np.dtype(dtype).kind in 'iu':
        max_value = (2 ** (np.dtype(dtype).itemsize * 8)) - 1
        color *= max_value
    color = np.append(color, max_value).astype(dtype)
    return color


def vertex_to_face_colors(vertex_colors, faces):
    '''
    将顶点颜色转换为面颜色

    :param vertex_colors: 顶点颜色数组
    :param faces: 面索引数组
    :return: 面颜色数组
    '''
    face_colors = vertex_colors[faces].mean(axis=2).astype(vertex_colors.dtype)
    return face_colors


def face_to_vertex_color(mesh, face_colors, dtype=COLOR_DTYPE):
    '''
    将一组面颜色转换为一组顶点颜色

    :param mesh: Trimesh 对象,表示网格
    :param face_colors: 面颜色数组
    :param dtype: 数据类型,默认为 COLOR_DTYPE
    :return: 顶点颜色数组
    '''
    color_dim = np.shape(face_colors)[1]

    vertex_colors = np.zeros((len(mesh.vertices), 3, color_dim))
    population = np.zeros((len(mesh.vertices), 3), dtype=bool)

    vertex_colors[[mesh.faces[:, 0], 0]] = face_colors
    vertex_colors[[mesh.faces[:, 1], 1]] = face_colors
    vertex_colors[[mesh.faces[:, 2], 2]] = face_colors

    population[[mesh.faces[:, 0], 0]] = True
    population[[mesh.faces[:, 1], 1]] = True
    population[[mesh.faces[:, 2], 2]] = True

    # 将总体和裁剪为1,以避免边界情况下的除法错误
    populated = np.clip(population.sum(axis=1), 1, 3)
    vertex_colors = vertex_colors.sum(axis=1) / populated.reshape((-1, 1))
    return vertex_colors.astype(dtype)
