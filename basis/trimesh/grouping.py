import numpy as np
from collections import deque
from networkx import from_edgelist, connected_components
from .util import decimal_to_digits, vector_to_spherical, spherical_to_vector, unitize
from .constants import log, tol

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    log.warning('Scipy不可用')


def merge_vertices_hash(mesh):
    '''
    删除重复的顶点,基于整数哈希

    这比在循环中查询KD树快大约20倍

    :param mesh: Trimesh对象
    '''
    unique, inverse = unique_rows(mesh.vertices)
    mesh.update_vertices(unique, inverse)


def merge_vertices_kdtree(mesh, angle=None):
    '''
    合并相同的顶点,即在彼此的笛卡尔距离TOL_MERGE内的顶点,然后替换mesh.faces中的引用

    :param mesh: Trimesh对象
    :param angle: float, 如果为None,则不考虑顶点法线
                  如果有值,只有在彼此的TOL_MERGE内且法线之间的角度小于angle_max时,才认为顶点相同

    性能注意:
    cKDTree需要scipy >= .12才能进行此查询类型
    你可能不想使用普通的python KDTree,因为它非常慢(在测试中慢约1000倍)
    '''
    tree = mesh.kdtree()
    used = np.zeros(len(mesh.vertices), dtype=bool)
    inverse = np.arange(len(mesh.vertices), dtype=int)
    unique = deque()
    vectors = mesh.vertex_normals
    for index, vertex in enumerate(mesh.vertices):
        if used[index]:
            continue
        neighbors = np.array(tree.query_ball_point(mesh.vertices[index], tol.merge))
        if angle is not None:
            groups = group_vectors(vectors[neighbors], angle=angle)[1]
        else:
            groups = np.arange(len(neighbors)).astype(int).reshape((1, -1))
        used[neighbors] = True
        for group in groups:
            inverse[neighbors[group]] = len(unique)
            unique.append(neighbors[group[0]])
    mesh.update_vertices(np.array(unique), inverse)


def replace_references(data, reference_dict):
    '''
    根据替换值的字典替换数组中的元素

    :param data: numpy数组
    :param reference_dict: 替换值映射的字典,例如:{2:1, 3:1, 4:5}
    '''
    shape = np.shape(data)
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view.reshape(shape)


def group(values, min_len=0, max_len=np.inf):
    '''
    返回相同值的索引组

    :param values: 1D 数组
    :param min_len: int,允许的最短组
                    所有组的长度将 >= min_len
    :param max_len: int,允许的最长组
                    所有组的长度将 <= max_len

    :return: 形成组的索引序列
             例如 [0,1,0,1] 返回 [[0,2], [1,3]]
    '''
    order = values.argsort()
    values = values[order]
    dupe = np.greater(np.abs(np.diff(values)), tol.zero)
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    dupe_len = np.diff(np.hstack((dupe_idx, len(values))))
    dupe_ok = np.logical_and(np.greater_equal(dupe_len, min_len), np.less_equal(dupe_len, max_len))
    groups = [order[i:(i + j)] for i, j in zip(dupe_idx[dupe_ok], dupe_len[dupe_ok])]
    return groups


def hashable_rows(data, digits=None):
    '''
    将数组转换为整数,基于给定的精度(由digits指定),然后将其放入可哈希的格式

    :param data: (n,m) 输入数组
    :param digits: 如果数据是浮点数,添加到哈希的位数
                   如果为None,将使用TOL_MERGE转换为位数
    :return: (n) 长度的自定义数据数组,可用于排序或作为哈希键
    '''
    as_int = float_to_int(data, digits)
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def float_to_int(data, digits=None):
    '''
    将numpy数组的数据表示为整数

    :param data: numpy数组
    :param digits: 位数
    :return: 整数表示的数组
    '''
    data = np.asanyarray(data)
    dtype_out = np.int32
    if data.size == 0:
        return data.astype(dtype_out)
    if digits is None:
        digits = decimal_to_digits(tol.merge)
    elif isinstance(digits, float) or isinstance(digits, float):
        digits = decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warn('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')
    # 如果数据已经是整数或布尔值,直接返回
    if data.dtype.kind in 'ib':
        as_int = data.astype(dtype_out)
    else:
        data_max = np.abs(data).max() * 10 ** digits
        dtype_out = [np.int32, np.int64][int(data_max > 2 ** 31)]
        as_int = (np.around(data, digits) * (10 ** digits)).astype(dtype_out)
    return as_int


def unique_ordered(data):
    '''
    返回与 np.unique相同的结果,但按数据中唯一值首次出现的顺序排序

    示例
    ---------
    In [1]: a = [0, 3, 3, 4, 1, 3, 0, 3, 2, 1]

    In [2]: np.unique(a)
    Out[2]: array([0, 1, 2, 3, 4])

    In [3]: trimesh.grouping.unique_ordered(a)
    Out[3]: array([0, 3, 4, 1, 2])
    '''
    data = np.asanyarray(data)
    order = np.sort(np.unique(data, return_index=True)[1])
    result = data[order]
    return result


def unique_float(data,
                 return_index=False,
                 return_inverse=False,
                 digits=None):
    '''
    类似于numpy.unique命令,但用于评估浮点数,使用指定的位数

    :param data: 输入数据数组
    :param return_index: 是否返回唯一值的索引
    :param return_inverse: 是否返回逆索引
    :param digits: 用于比较的位数.如果未指定,将使用库默认的TOL_MERGE
    :return: 唯一值数组,可能还包括索引和逆索引
    '''
    as_int = float_to_int(data, digits)
    _junk, unique, inverse = np.unique(as_int, return_index=True, return_inverse=True)
    if (not return_index) and (not return_inverse):
        return data[unique]
    result = [data[unique]]
    if return_index:   result.append(unique)
    if return_inverse: result.append(inverse)
    return tuple(result)


def unique_rows(data, digits=None):
    '''
    返回唯一行的索引.它将返回重复行的首次出现

    [[1,2], [3,4], [1,2]] 将返回 [0,1]

    :param data: (n,m) 浮点数据集
    :param digits: 用于唯一性比较的位数
    :return:
    - unique: (j) 数组,数据中唯一行的索引
    - inverse: (n) 长度数组,用于重构原始数据
               例如:unique[inverse] == data
    '''
    hashes = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    return unique, inverse


def unique_value_in_row(data, unique=None):
    '''
    对于一个二维整数数组,找到每行中只出现一次的值的位置
    如果每行中有多个只出现一次的值,则返回最后一个

    :param data: (n,d) 整数数组
    :param unique: (m) 整数,数据中包含的唯一值列表.
                   仅用于加速,如果未传递,将从np.unique生成
    :return: (n,d) 布尔数组,每行有一个或零个True值

    Example
    -------------------------------------
    In [0]: r = np.array([[-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1, -1]], dtype=np.int8)

    In [1]: unique_value_in_row(r)
    Out[1]: 
           array([[ True, False, False],
                  [False,  True, False],
                  [ True, False, False],
                  [False,  True, False],
                  [False,  True, False]], dtype=bool)

    In [2]: unique_value_in_row(r).sum(axis=1)
    Out[2]: array([1, 1, 1, 1, 1])

    In [3]: r[unique_value_in_row(r)]
    Out[3]: array([-1,  1, -1,  1,  1], dtype=int8)
    '''
    if unique is None:
        unique = np.unique(data)
    data = np.asanyarray(data)
    result = np.zeros_like(data, dtype=bool, subok=False)
    for value in unique:
        test = np.equal(data, value)
        test_ok = test.sum(axis=1) == 1
        result[test_ok] = test[test_ok]
    return result


def group_rows(data, require_count=None, digits=None):
    '''
    返回重复行的索引组

    例如:
    [[1,2], [3,4], [1,2]] 将返回 [[0,2], [1]]

    :param data: (n,m) 数组
    :param require_count: 仅返回指定长度的组,例如:
                          require_count =  2
                          [[1,2], [3,4], [1,2]] 将返回 [[0,2]]

                          注意,使用require_count允许使用numpy高级索引
                          代替循环和检查哈希,因此速度快约10倍.
    :param digits: 如果数据是浮点数,考虑多少位小数.
                   如果为None,将使用TOL_MERGE转换为位数.

    :return:
    - groups: 指示相同行的数据索引的列表或序列.
              如果require_count != None,形状为 (j, require_count)
              如果require_count为None,形状将不规则(即序列)
    '''

    def group_dict():
        '''
        基于简单哈希表的分组

        由于循环和附加操作,在非常大的数组上这会比较慢
        但它适用于不规则的组
        '''
        observed = dict()
        hashable = hashable_rows(data, digits=digits)
        for index, key in enumerate(hashable):
            key_string = key.tostring()
            if key_string in observed:
                observed[key_string].append(index)
            else:
                observed[key_string] = [index]
        return np.array(list(observed.values()))

    def group_slice():
        # 创建可以排序的行表示
        hashable = hashable_rows(data, digits=digits)
        # 记录行的顺序,以便稍后可以返回原始索引
        order = np.argsort(hashable)
        # 现在,我们希望我们的哈希值已排序
        hashable = hashable[order]
        # 检查每个相邻元素是否相等
        # example: hashable = [1, 1, 1]; dupe = [0, 0]
        dupe = hashable[1:] != hashable[:-1]
        # 我们希望获得组的第一个索引,以便可以从该位置进行切片
        # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
        dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
        # 如果您希望使用此一个函数处理非规则组
        # 您可以使用:np.array_split(dupe_idx)
        # 这比上面的group_dict方法慢大约3倍.
        start_ok = np.diff(np.hstack((dupe_idx, len(hashable)))) == require_count
        groups = np.tile(dupe_idx[start_ok].reshape((-1, 1)),
                         require_count) + np.arange(require_count)
        groups_idx = order[groups]
        if require_count == 1:
            return groups_idx.reshape(-1)
        return groups_idx

    if require_count is None:
        return group_dict()
    else:
        return group_slice()


def boolean_rows(a, b, operation=set.intersection):
    '''
    找到两个数组中同时出现的行

    :param a: (n,d) 数组
    :param b: (m,d) 数组
    :param operation: 布尔集合操作函数:
                      set.intersection
                      set.difference
                      set.union
    :return: shared: (p, d) 数组,包含a和b中同时出现的行
    '''
    a = float_to_int(a)
    b = float_to_int(b)
    shared = operation({tuple(i) for i in a}, {tuple(j) for j in b})
    shared = np.array(list(shared))
    return shared


def group_vectors(vectors,
                  angle=np.radians(10),
                  include_negative=False):
    '''
    基于角度公差对向量进行分组,并可选择包括负向量

    这与group_rows(stack_negative(rows))非常相似
    主要区别在于max_angle可以更宽松,因为我们正在进行实际的距离查询
    '''
    dist_max = np.tan(angle)
    unit_vectors, valid = unitize(vectors, check_valid=True)
    valid_index = np.nonzero(valid)[0]
    consumed = np.zeros(len(unit_vectors), dtype=bool)
    tree = KDTree(unit_vectors)
    unique_vectors = deque()
    aligned_index = deque()

    for index, vector in enumerate(unit_vectors):
        if consumed[index]:
            continue
        aligned = np.array(tree.query_ball_point(vector, dist_max))
        vectors = unit_vectors[aligned]
        if include_negative:
            aligned_neg = tree.query_ball_point(-1.0 * vector, dist_max)
            vectors = np.vstack((vectors, -unit_vectors[aligned_neg]))
            aligned = np.append(aligned, aligned_neg)
        aligned = aligned.astype(int)
        consumed[aligned] = True
        unique_vectors.append(np.median(vectors, axis=0))
        aligned_index.append(valid_index[aligned])
    return np.array(unique_vectors), np.array(aligned_index)


def group_vectors_spherical(vectors, angle=np.radians(10)):
    '''
    基于角度公差对向量进行分组,并可选择包括负向量

    这与group_rows(stack_negative(rows))非常相似
    主要区别在于max_angle可以更宽松,因为我们正在进行实际的距离查询
    '''
    spherical = vector_to_spherical(vectors)
    angles, groups = group_distance(spherical, angle)
    new_vectors = spherical_to_vector(angles)
    return new_vectors, groups


def group_distance(values, distance):
    """
    根据给定的距离对值进行分组

    :param values: (n, d) 数组,表示要分组的值
    :param distance: float,表示分组的最大距离
    :return: unique: (n, d) 数组,表示唯一的值
             groups: (n) 序列,表示值的索引
    """
    consumed = np.zeros(len(values), dtype=bool)
    tree = KDTree(values)
    # (n, d) 集合,表示唯一的值
    unique = deque()
    # (n) 序列,表示值的索引
    groups = deque()
    for index, value in enumerate(values):
        if consumed[index]:
            continue
        group = np.array(tree.query_ball_point(value, distance), dtype=int)
        consumed[group] = True
        unique.append(np.median(values[group], axis=0))
        groups.append(group)
    return np.array(unique), np.array(groups)


def stack_negative(rows):
    """
    给定输入行 (n, d),返回一个 (n, 2*d) 的数组
    该数组是符号无关的
    """
    rows = np.asanyarray(rows)
    stacked = np.column_stack((rows, rows * -1))
    nonzero = np.abs(rows) > tol.zero
    negative = rows < -tol.zero
    roll = np.logical_and(nonzero, negative).any(axis=1)
    stacked[roll] = np.roll(stacked[roll], 3, axis=1)
    return stacked


def clusters(points, radius):
    """
    找到距离小于给定半径的点簇

    :param points: nxd 数组,表示点
    :param radius: float,表示簇中点之间的最大距离
    :return: [point_list, ...],表示点簇的列表

    author: reviserd by weiwei
    date: 20210120
    """
    tree = KDTree(points)
    pairs = tree.query_pairs(radius)
    graph = from_edgelist(pairs)
    groups = list(connected_components(graph))
    return groups


def blocks(data, min_len=2, max_len=np.inf, digits=None):
    '''
    给定一个数组,找到相同值的连续块的索引

    :param data: (n) 数组
    :param min_len: int,返回的最小长度组
    :param max_len: int,返回的最大长度组
    :param digits: 如果处理浮点数,使用多少位数

    :return: blocks: (m) 序列,引用数据的索引
    '''
    data = float_to_int(data, digits=digits)
    # 找到拐点,或数组从 True 转为 False 的位置
    infl = np.hstack(([0],
                      np.nonzero(np.diff(data))[0] + 1,
                      [len(data)]))
    infl_len = np.diff(infl)
    infl_ok = np.logical_and(infl_len >= min_len,
                             infl_len <= max_len)
    if data.dtype.kind == 'b':
        # 通过检查每个连续块的第一个值,确保每个块的值为 True
        infl_ok = np.logical_and(infl_ok,
                                 data[infl[:-1]])
    # 将开始/结束索引扩展为完整的值范围
    blocks = [np.arange(infl[i], infl[i + 1]) for i, ok in enumerate(infl_ok) if ok]
    return blocks
