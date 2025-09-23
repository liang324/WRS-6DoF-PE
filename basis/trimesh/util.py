# trimesh.Util:实用函数 这个文件只允许从numpy和标准库中导入

import numpy as np
import logging
import hashlib
import base64
from collections import defaultdict, deque
from sys import version_info

if version_info.major >= 3:
    basestring = str

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

# 包含在这里所以util只导入了标准库
_TOL_ZERO = 1e-12


def unitize(points, check_valid=False):
    '''
    将向量列表转换为单位向量列表

    :param points: (n,m) 或 (j) 输入向量数组.对于1D数组,points被视为单个向量；对于2D数组,每行被视为一个向量.
    :param check_valid: 布尔值,如果为True,则启用有效输出和检查.
    :return: (n,m) 或 (j) 长度的单位向量数组.如果check_valid为True,还返回一个布尔数组,指示哪些向量是有效的(非零长度).
    '''
    points = np.asanyarray(points)
    axis = len(points.shape) - 1
    length = np.sum(points ** 2, axis=axis) ** .5

    if check_valid:
        valid = np.greater(length, _TOL_ZERO)
        # 如果是二维数组
        if axis == 1:
            unit_vectors = (points[valid].T / length[valid]).T
        # 如果是一维且有效
        elif len(points.shape) == 1 and valid:
            unit_vectors = points / length
        else:
            unit_vectors = np.array([])
        return unit_vectors, valid
    else:
        # 添加零值检查
        length[length < _TOL_ZERO] = 1.0  # 将零长度替换为 1,避免除以零
        unit_vectors = (points.T / length).T
    return unit_vectors


# def unitize(points, check_valid=False):
#     '''
#     将向量列表转换为单位向量列表
#
#     :param points: (n,m) 或 (j) 输入向量数组.对于1D数组,points被视为单个向量；对于2D数组,每行被视为一个向量.
#     :param check_valid: 布尔值,如果为True,则启用有效输出和检查.
#     :return: (n,m) 或 (j) 长度的单位向量数组.如果check_valid为True,还返回一个布尔数组,指示哪些向量是有效的(非零长度).
#     '''
#     points = np.asanyarray(points)
#     axis = len(points.shape) - 1
#     length = np.sum(points ** 2, axis=axis) ** .5
#
#     if check_valid:
#         valid = np.greater(length, _TOL_ZERO)
#         # 如果是二维数组
#         if axis == 1:
#             unit_vectors = (points[valid].T / length[valid]).T
#         # 如果是一维且有效
#         elif len(points.shape) == 1 and valid:
#             unit_vectors = points / length
#         else:
#             unit_vectors = np.array([])
#         return unit_vectors, valid
#     else:
#         if np.isscalar(length):  # 如果是单个向量(标量)
#             if length < _TOL_ZERO:
#                 length = 1.0
#             unit_vectors = points / length
#         else:  # 批量向量
#             length[length < _TOL_ZERO] = 1.0
#             unit_vectors = (points.T / length).T
#     return unit_vectors


def transformation_2D(offset=[0.0, 0.0], theta=0.0):
    '''
    生成2D齐次变换矩阵

    :param offset: [float, float],平移偏移量
    :param theta: float,旋转角度(弧度)
    :return: 3x3的2D齐次变换矩阵
    '''
    T = np.eye(3)
    s = np.sin(theta)
    c = np.cos(theta)

    T[0, 0:2] = [c, s]
    T[1, 0:2] = [-s, c]
    T[0:2, 2] = offset
    return T


def euclidean(a, b):
    '''
    计算向量a和b之间的欧几里得距离

    :param a: 数组或列表,向量a
    :param b: 数组或列表,向量b
    :return: float,向量a和b之间的欧几里得距离
    '''
    return np.sum((np.array(a) - b) ** 2) ** .5


def is_file(obj):
    '''
    判断对象是否为文件

    :param obj: 任意对象
    :return: 布尔值,True表示对象是文件
    '''
    return hasattr(obj, 'read')


def is_string(obj):
    '''
    判断对象是否为字符串

    :param obj: 任意对象
    :return: 布尔值,True表示对象是字符串
    '''
    return isinstance(obj, basestring)


def is_dict(obj):
    '''
    判断对象是否为字典

    :param obj: 任意对象
    :return: 布尔值,True表示对象是字典
    '''
    return isinstance(obj, dict)


def is_sequence(obj):
    '''
    判断对象是否为序列

    :param obj: 任意对象
    :return: 布尔值,True表示对象是序列
    '''
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))
    seq = seq and not isinstance(obj, dict)
    # numpy有时会返回单个float64类型的对象,但是看起来像序列,所以我们检查形状
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()
    return seq


def is_shape(obj, shape):
    '''
    比较 numpy.ndarray 的形状与目标形状,任何小于零的值被视为通配符

    :param obj: np.ndarray,检查形状的对象
    :param shape: 列表或元组,目标形状.任何负数项将被视为通配符,任何元组项将被视为OR
    :return: 布尔值,True表示对象的形状与查询形状匹配
    '''
    if (not hasattr(obj, 'shape') or
            len(obj.shape) != len(shape)):
        return False

    for i, target in zip(obj.shape, shape):
        # 检查当前字段是否有多个可接受的值
        if is_sequence(target):
            if i in target:
                continue
            else:
                return False
        # 检查current字段是否为通配符
        if target < 0:
            if i == 0:
                return False
            else:
                continue
        # 由于我们只有一个目标和一个值,如果它们不相等,我们有答案
        if target != i:
            return False

    # 由于没有一个检查失败,所以这两个形状是相同的
    return True


def make_sequence(obj):
    '''
    给定一个对象,如果它是一个序列则返回,否则将其添加到长度为1的序列中并返回

    :param obj: 任意对象
    :return: np.array,包含对象的数组
    '''
    if is_sequence(obj):
        return np.array(obj)
    else:
        return np.array([obj])


def vector_to_spherical(cartesian):
    '''
    将一组笛卡尔点转换为 (n,2) 球面向量

    :param cartesian: (n,3) 笛卡尔坐标数组
    :return: (n,2) 球面坐标数组
    '''
    x, y, z = np.array(cartesian).T
    # 在除零错误上作弊
    x[np.abs(x) < _TOL_ZERO] = _TOL_ZERO
    if np.any(x == 0):
        print("Warning: x array contains zero values!")
    spherical = np.column_stack((np.arctan(y / x), np.arccos(z)))
    return spherical


def spherical_to_vector(spherical):
    """
    将一组 nx2 球面向量转换为 nx3 向量

    :param spherical: (n,2) 球面坐标数组
    :return: (n,3) 笛卡尔坐标数组

    author: revised by weiwei
    date: 20210120
    """
    spherical = np.asanyarray(spherical, dtype=np.float64)
    if not is_shape(spherical, (-1, 2)):
        raise ValueError('spherical coordinates must be (n, 2)!')
    theta, phi = spherical.T
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    vectors = np.column_stack((ct * sp,
                               st * sp,
                               cp))
    return vectors


def diagonal_dot(a, b):
    '''
    按行计算 a 和 b 的点积

    :param a: (n,m) 数组
    :param b: (n,m) 数组
    :return: (n,) 数组,表示每行的点积
    '''
    result = (np.array(a) * b).sum(axis=1)
    return result


def three_dimensionalize(points, return_2D=True):
    '''
    给定一组 (n,2) 或 (n,3) 点,将它们返回为 (n,3) 点

    :param points: (n, 2) 或 (n,3) 点数组
    :param return_2D: 布尔标志,指示是否返回二维信息
    :return: 如果 return_2D 为 True,返回 (is_2D, points),否则返回 points
    '''
    points = np.asanyarray(points)
    shape = points.shape
    if len(shape) != 2:
        raise ValueError('Points must be 2D array!')
    if shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
        is_2D = True
    elif shape[1] == 3:
        is_2D = False
    else:
        raise ValueError('Points must be (n,2) or (n,3)!')
    if return_2D:
        return is_2D, points
    return points


def grid_arange_2D(bounds, step):
    '''
    返回具有指定间距的二维网格

    :param bounds: (2,2) 列表,表示 [[minx, miny], [maxx, maxy]]
    :param step: float,点之间的间隔
    :return: (n, 2) 二维点列表
    '''
    x_grid = np.arange(*bounds[:, 0], step=step)
    y_grid = np.arange(*bounds[:, 1], step=step)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def grid_linspace_2D(bounds, count):
    '''
    返回一个 count*count 的二维网格

    :param bounds: (2,2) 列表,表示 [[minx, miny], [maxx, maxy]]
    :param count: int,边上的元素数量
    :return: (count**2, 2) 二维点列表
    '''
    x_grid = np.linspace(*bounds[:, 0], count=count)
    y_grid = np.linspace(*bounds[:, 1], count=count)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def replace_references(data, reference_dict):
    """
    替换数据中的引用

    :param data: 数据列表或数组
    :param reference_dict: 字典,包含要替换的引用
    :return: 替换后的数据视图
    """
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view


def multi_dict(pairs):
    '''
    给定一组键值对,创建一个字典.如果一个键出现多次,将值堆叠到一个数组中

    :param pairs: (n,2) 键值对数组
    :return: 字典,存储所有值(而不是常规字典中的最后一个)
    '''
    result = defaultdict(list)
    for k, v in pairs:
        result[k].append(v)
    return result


def tolist_dict(data):
    '''
    将字典中的所有值转换为列表

    :param data: 字典
    :return: 转换后的字典
    '''

    def tolist(item):
        if hasattr(item, 'tolist'):
            return item.tolist()
        else:
            return item

    result = {k: tolist(v) for k, v in data.items()}
    return result


# 胡老师代码
def is_binary_file(file_obj, probe_sz=1024):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    try:
        start = file_obj.tell()
        fbytes = file_obj.read(probe_sz)
        file_obj.seek(start)
        is_str = isinstance(fbytes, str)
        for fbyte in fbytes:
            if is_str:
                code = ord(fbyte)
            else:
                code = fbyte
            if code > 127: return True
    except UnicodeDecodeError:
        return True
    return False

#
# # 陈老师代码
# def is_binary_file(file_obj):
#     '''
#     如果文件包含非 ASCII 字符(> 0x7F 或 127),返回 True
#
#     :param file_obj: 文件对象
#     :return: 布尔值,指示文件是否为二进制文件
#     '''
#     start = file_obj.tell()
#     fbytes = file_obj.read(1024)
#     file_obj.seek(start)
#     is_str = isinstance(fbytes, str)
#     for fbyte in fbytes:
#         if is_str:
#             code = ord(fbyte)
#         else:
#             code = fbyte
#         if code > 127: return True
#     return False


def decimal_to_digits(decimal, min_digits=None):
    '''
    将小数转换为数字位数

    :param decimal: float,小数
    :param min_digits: int,最小位数
    :return: int,数字位数
    '''
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def md5_object(obj):
    '''
    如果对象是可哈希的,返回 MD5 的十六进制字符串

    :param obj: 可哈希对象
    :return: MD5 哈希的十六进制字符串
    '''
    hasher = hashlib.md5()
    hasher.update(obj)
    hashed = hasher.hexdigest()
    return hashed


def attach_to_log(log_level=logging.DEBUG, blacklist=['TerminalIPythonApp', 'PYREADLINE']):
    '''
    将流处理器附加到所有记录器

    :param log_level: 日志级别
    :param blacklist: 列表,包含不附加处理器的记录器名称
    '''
    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            ("%(log_color)s%(levelname)-8s%(reset)s " +
             "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
            datefmt=None,
            reset=True,
            log_colors={'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red'})
    except ImportError:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(log_level)

    for logger in logging.Logger.manager.loggerDict.values():
        if (logger.__class__.__name__ != 'Logger' or
                logger.name in blacklist):
            continue
        logger.addHandler(handler_stream)
        logger.setLevel(log_level)
    np.set_printoptions(precision=5, suppress=True)


def tracked_array(array, dtype=None):
    '''
    正确地子类化一个 numpy ndarray 以跟踪更改

    :param array: 数组
    :param dtype: 数据类型
    :return: TrackedArray 对象
    '''
    result = np.ascontiguousarray(array).view(TrackedArray)
    if dtype is None:
        return result
    return result.astype(dtype)


class TrackedArray(np.ndarray):
    '''
    跟踪 numpy ndarray 中的更改

    md5: 返回数组的 MD5 的十六进制字符串
    '''

    def __array_finalize__(self, obj):
        '''
        在每个 TrackedArray 上设置一个修改标志
        这个标志将在每次更改时设置,以及在复制和某些类型的切片期间设置
        '''
        self._modified = True
        if isinstance(obj, type(self)):
            obj._modified = True

    def md5(self):
        '''
        以十六进制字符串形式返回当前数组的MD5散列值.
        这是相当快的；在现代的i7桌面上,a(1000000,3)浮点数
        数组在0.03秒内可靠地散列
        只有在设置了可能为false的修改标志时才会重新计算
        positive(强制进行不必要的重算),但不包含false
        negative,返回错误的哈希值
        '''
        if self._modified or not hasattr(self, '_hashed'):
            self._hashed = md5_object(self)
        self._modified = False
        return self._hashed

    def __hash__(self):
        '''
        哈希需要返回一个整数,因此我们将十六进制字符串转换为整数
        '''
        return int(self.md5(), 16)

    def __setitem__(self, i, y):
        """
        设置数组项并标记为已修改
        """
        self._modified = True
        super(self.__class__, self).__setitem__(i, y)

    def __setslice__(self, i, j, y):
        '''
        设置数组切片并标记为已修改
        '''
        self._modified = True
        super(self.__class__, self).__setslice__(i, j, y)


class Cache:
    """
    用于缓存值的类,直到 id 函数发生变化
    """

    def __init__(self, id_function=None):
        """
        初始化缓存类

        :param id_function: 一个函数,用于生成唯一标识符.如果未提供,则使用默认函数
        """
        if id_function is None:
            self._id_function = lambda: None
        else:
            self._id_function = id_function
        self.id_current = None
        self._lock = 0
        self.cache = {}

    def decorator(self, function):
        """
        装饰器方法,用于缓存函数的结果

        :param function: 要缓存结果的函数
        :return: 缓存的结果
        """
        name = function.__name__
        if name in self.cache:
            return self.cache[name]
        result = function()
        self.cache[name] = result
        return result

    def get(self, key):
        """
        从缓存中获取一个键
        如果键不可用或缓存已失效,则返回 None

        :param key: 要获取的键
        :return: 缓存的值或 None
        author: revised by weiwei
        date: 20201201
        """
        self.verify()
        if key in self.cache:
            return self.cache[key]
        return None

    def verify(self):
        """
        验证缓存的值是否仍然对应于相同的 id_function 值
        如果 id_function 的值已更改,则删除所有存储的项目

        author: revised by weiwei
        date: 20201201
        """
        id_new = self._id_function()
        if (self._lock == 0) and (id_new != self.id_current):
            if len(self.cache) > 0:
                log.debug('%d items cleared from cache: %s',
                          len(self.cache),
                          str(self.cache.keys()))
            self.clear()
            self.id_set()

    def clear(self, exclude=None):
        """
        删除缓存中的所有元素

        :param exclude: 要排除的键列表

        author: revised by weiwei
        date: 20201201
        """
        if exclude is None:
            self.cache = {}
        else:
            self.cache = {k: v for k, v in self.cache.items() if k in exclude}

    def update(self, items):
        """
        使用一组键值对更新缓存,而不检查 id_function

        :param items: 要更新的键值对

        author: revised by weiwei
        date: 20201201
        """
        self.cache.update(items)
        self.id_set()

    def id_set(self):
        """
        设置当前 id 为 id_function 的返回值
        """
        self.id_current = self._id_function()

    def set(self, key, value):
        """
        设置缓存中的键值对

        :param key: 要设置的键
        :param value: 要设置的值
        :return: 设置的值
        """
        self.verify()
        self.cache[key] = value
        return value

    def __getitem__(self, key):
        """
        获取缓存中的键值

        :param key: 要获取的键
        :return: 缓存的值或 None
        """
        return self.get(key)

    def __setitem__(self, key, value):
        """
        设置缓存中的键值对

        :param key: 要设置的键
        :param value: 要设置的值
        :return: 设置的值
        """
        return self.set(key, value)

    def __contains__(self, key):
        """
        检查缓存中是否包含指定的键

        :param key: 要检查的键
        :return: 布尔值,指示键是否在缓存中
        """
        self.verify()
        return key in self.cache

    def __enter__(self):
        """
        进入上下文管理器,增加锁计数
        """
        self._lock += 1

    def __exit__(self, *args):
        """
        退出上下文管理器,减少锁计数并更新当前 id
        """
        self._lock -= 1
        self.id_current = self._id_function()


class DataStore:
    """
    一个用于存储和管理数据的类
    """

    def __init__(self):
        """
        初始化 DataStore 类,创建一个空的数据字典
        """
        self.data = {}

    @property
    def mutable(self):
        """
        获取或设置数据的可变性

        :return: 布尔值,指示数据是否可变
        """
        if not hasattr(self, '_mutable'):
            self._mutable = True
        return self._mutable

    @mutable.setter
    def mutable(self, value):
        """
        设置数据的可变性

        :param value: 布尔值,指示数据是否应设置为可变
        """
        value = bool(value)
        for i in self.data.value():
            i.flags.writeable = value
        self._mutable = value

    def is_empty(self):
        """
        检查数据存储是否为空

        :return: 布尔值,True 表示数据存储为空
        """
        if len(self.data) == 0:
            return True
        for v in self.data.values():
            if is_sequence(v):
                if len(v) > 0:
                    return False
            else:
                if bool(np.isreal(v)):
                    return False
        return True

    def clear(self):
        """
        清空数据存储
        """
        self.data = {}

    def __getitem__(self, key):
        """
        获取指定键的数据

        :param key: 要获取数据的键
        :return: 对应键的数据,如果键不存在则返回 None
        """
        try:
            return self.data[key]
        except KeyError:
            return None

    def __setitem__(self, key, data):
        """
        设置指定键的数据

        :param key: 要设置数据的键
        :param data: 要存储的数据
        """
        self.data[key] = tracked_array(data)

    def __len__(self):
        """
        获取数据存储的长度

        :return: 数据存储中的项数
        """
        return len(self.data)

    def values(self):
        """
        获取数据存储中的所有值

        :return: 数据存储中的值的集合
        """
        return self.data.values()

    def md5(self):
        """
        计算数据存储的 MD5 哈希值

        :return: 数据存储的 MD5 哈希值
        """
        md5 = ''
        for key in np.sort(list(self.data.keys())):
            md5 += self.data[key].md5()
        return md5


def stack_lines(indices):
    """
    将索引转换为线段的起点和终点

    :param indices: 索引数组
    :return: (n,2) 线段的起点和终点数组
    """
    return np.column_stack((indices[:-1], indices[1:])).reshape((-1, 2))


def append_faces(vertices_seq, faces_seq):
    '''
    给定一系列零索引的面和顶点,将它们合并为一个 (n,3) 的面列表和 (m,3) 的顶点列表.

    :param vertices_seq: (n) 顶点数组序列
    :param faces_seq: (n) 面数组序列,零索引并引用其对应的顶点
    :return: 合并后的顶点和面数组
    '''
    vertices_len = np.array([len(i) for i in vertices_seq])
    face_offset = np.append(0, np.cumsum(vertices_len)[:-1])

    for offset, faces in zip(face_offset, faces_seq):
        faces += offset

    vertices = np.vstack(vertices_seq)
    faces = np.vstack(faces_seq)

    return vertices, faces


def array_to_encoded(array, dtype=None, encoding='base64'):
    '''
    将 numpy 数组导出为紧凑的可序列化字典

    :param array: numpy 数组
    :param dtype: 可选,数组编码使用的数据类型
    :param encoding: str,'base64' 或 'binary'
    :return: 编码后的字典,包含 dtype、shape 和 base64 编码的字符串
    '''
    array = np.asanyarray(array)
    shape = array.shape
    # 拉威尔也强制连续
    flat = np.ravel(array)
    if dtype is None:
        dtype = array.dtype
    encoded = {'dtype': np.dtype(dtype).str, 'shape': shape}
    if encoding in ['base64', 'dict64']:
        packed = base64.b64encode(flat.astype(dtype))
        if hasattr(packed, 'decode'):
            packed = packed.decode('utf-8')
        encoded['base64'] = packed
    elif encoding == 'binary':
        encoded['binary'] = array.tostring(order='C')
    else:
        raise ValueError('encoding {} is not available!'.format(encoding))
    return encoded


def encoded_to_array(encoded):
    '''
    将包含 base64 编码字符串的字典转换回 numpy 数组

    :param encoded: 编码字典,包含 dtype、shape 和 base64 编码的字符串
    :return: numpy 数组
    '''
    shape = encoded['shape']
    dtype = np.dtype(encoded['dtype'])
    if 'base64' in encoded:
        array = np.fromstring(base64.b64decode(encoded['base64']), dtype).reshape(shape)
    elif 'binary' in encoded:
        array = np.fromstring(encoded['binary'], dtype=dtype, count=np.product(shape))
    array = array.reshape(shape)
    return array


def is_instance_named(obj, name):
    '''
    给定一个对象,如果它是类 'name' 的成员或子类,则返回 True

    :param obj: 类的实例
    :param name: 字符串,类名
    :return: 布尔值,指示对象是否是指定类的成员
    '''
    try:
        type_named(obj, name)
        return True
    except ValueError:
        return False


def type_bases(obj, depth=4):
    '''
    返回传入对象的基类

    :param obj: 对象
    :param depth: 搜索深度
    :return: 基类数组
    '''
    bases = deque([list(obj.__class__.__bases__)])
    for i in range(depth):
        bases.append([i.__base__ for i in bases[-1] if i is not None])
    try:
        bases = np.hstack(bases)
    except IndexError:
        bases = []
    # 我们使用hasattr,因为None/NoneType可以包含在基列表中
    bases = [i for i in bases if hasattr(i, '__name__')]
    return np.array(bases)


def type_named(obj, name):
    '''
    类似于内置的 `type()`,但在类的基类中查找指定名称的实例

    :param obj: 要查找类的对象
    :param name: str,类的名称
    :return: 指定名称的类,或 None
    '''
    # 如果obj是指定类的成员,返回True
    name = str(name)
    if obj.__class__.__name__ == name:
        return obj.__class__
    for base in type_bases(obj):
        if base.__name__ == name:
            return base
    raise ValueError('Unable to extract class of name ' + name)


def concatenate(a, b=None):
    """
    连接两个网格

    :param a: Trimesh 对象
    :param b: Trimesh 对象
    :return: 包含 a 和 b 所有面的 Trimesh 对象

    author: weiwei
    date: 20210120
    """
    if b is None:
        b = []
        # 将网格堆叠到平面列表中
    meshes = np.append(a, b)
    # 如果只有一个网格,就返回第一个
    if len(meshes) == 1:
        return meshes[0].copy()
    # 提取trimesh类型以避免循环导入,并断言两个输入都是Trimesh对象
    trimesh_type = type_named(meshes[0], 'Trimesh')

    # 添加网格的面和顶点
    vertices, faces = append_faces([m.vertices.copy() for m in meshes], [m.faces.copy() for m in meshes])
    # 仅保存已计算的人脸法线
    face_normals = None
    if all('face_normals' in m._cache for m in meshes):
        face_normals = np.vstack([m.face_normals for m in meshes])
    try:
        # 拼接视觉效果
        visual = meshes[0].visual.concatenate([m.visual for m in meshes[1:]])
    except BaseException:
        log.warning('failed to combine visuals', exc_info=True)
        visual = None
    # 创建mesh对象
    mesh = trimesh_type(vertices=vertices,
                        faces=faces,
                        face_normals=face_normals,
                        visual=visual,
                        process=False)
    return mesh


def submesh(mesh,
            faces_sequence,
            only_watertight=False,
            append=False):
    '''
    返回网格的子集

    :param mesh: Trimesh 对象
    :param faces_sequence: 网格中的面索引序列
    :param only_watertight: 仅返回水密的子网格
    :param append: 返回一个包含指定面的单一网格,如果设置此标志,则忽略 only_watertight
    :return: 如果 append 为 True,返回 Trimesh 对象；否则返回 Trimesh 对象的列表
    '''
    # 避免在原始网格上破坏缓存
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = deque()
    vertices = deque()
    normals = deque()
    visuals = deque()

    # 对人脸重建驯鹿
    mask = np.arange(len(original_vertices))

    for faces_index in faces_sequence:
        # 清理索引,以防它们以集合或元组的形式传入
        faces_index = np.array(list(faces_index))
        faces_current = original_faces[faces_index]
        unique = np.unique(faces_current.reshape(-1))

        # 从0重新定义面索引
        mask[unique] = np.arange(len(unique))

        normals.append(mesh.face_normals[faces_index])
        faces.append(mask[faces_current])
        vertices.append(original_vertices[unique])
        visuals.extend(mesh.visual.subsets([faces_index]))

    # 我们使用type(mesh)而不是从base导入Trimesh,因为这会导致循环导入
    trimesh_type = type_named(mesh, 'Trimesh')
    if append:
        visuals = np.array(visuals)
        vertices, faces = append_faces(vertices, faces)
        appended = trimesh_type(vertices=vertices,
                                faces=faces,
                                face_normals=np.vstack(normals),
                                visual=visuals[0].union(visuals[1:]),
                                process=False)
        return appended
    result = [trimesh_type(vertices=v,
                           faces=f,
                           face_normals=n,
                           visual=c,
                           process=False) for v, f, n, c in zip(vertices,
                                                                faces,
                                                                normals,
                                                                visuals)]
    result = np.array(result)
    if only_watertight:
        watertight = np.array([i.fill_holes() and len(i.faces) > 4 for i in result])
        result = result[watertight]
    return result


def zero_pad(data, count, right=True):
    '''
    对数据进行零填充

    :param data: (n) 长度的一维数组
    :param count: int,目标长度
    :return: 如果 (n < count),返回长度为 count 的一维数组,否则返回长度为 (n) 的数组
    '''
    if len(data) == 0:
        return np.zeros(count)
    elif len(data) < count:
        padded = np.zeros(count)
        if right:
            padded[-len(data):] = data
        else:
            padded[:len(data)] = data
        return padded
    else:
        return np.asanyarray(data)


def format_json(data, digits=6):
    '''
    将一维浮点数组转换为 JSON 字符串

    :param data: (n,) 浮点数组
    :param digits: int,浮点数的小数位数
    :return: as_json: str,格式化为 JSON 可解析字符串的数据
    '''
    format_str = '.' + str(int(digits)) + 'f'
    as_json = '[' + ','.join(map(lambda o: format(o, format_str), data)) + ']'
    return as_json


class Words:
    '''
    包含单词列表的类,例如英语单词.主要目的是创建随机短语,用于命名事物,而不需要使用巨大的哈希字符串
    '''

    def __init__(self, file_name='/usr/share/dict/words', words=None):
        '''
        初始化 Words 类

        :param file_name: str,包含单词的文件路径
        :param words: list,单词列表
        '''
        if words is None:
            self.words = np.loadtxt(file_name, dtype=str)
        else:
            self.words = np.array(words, dtype=str)

        self.words_simple = np.array([i.lower() for i in self.words if str.isalpha(i)])
        if len(self.words) == 0:
            log.warning('No words available!')

    def random_phrase(self, length=2, delimiter='-'):
        '''
        使用仅包含字符的单词创建随机短语

        :param length: int,短语中的单词数量
        :param delimiter: str,分隔单词的字符
        :return: phrase: str,长度为 length 的短语,单词之间用 delimiter 分隔
        '''
        result = str(delimiter).join(np.random.choice(self.words_simple, length))
        return result


def grid_linspace(bounds, count):
    """
    返回一个在边界框内间隔的网格,边缘使用 np.linspace 间隔

    :param bounds: (2, dimension) 列表,格式为 [[min x, min y, ...], [max x, max y, ...]]
    :param count: int 或 (dimension,) int,每边的样本数量
    :return: grid: (n, dimension) 浮点数,指定边界内的点
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError('bounds must be (2, dimension!')
    count = np.asanyarray(count, dtype=np.int64)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])
    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing='ij')
                     ).reshape(bounds.shape[1], -1).T
    return grid
