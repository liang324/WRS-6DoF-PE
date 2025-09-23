import numpy as np

from .transformations import rotation_matrix
from .constants import tol, log
from .util import unitize, stack_lines

try:
    from scipy.sparse import coo_matrix
except ImportError:
    log.warning('scipy.sparse.coo_matrix unavailable')


def plane_transform(origin, normal):
    '''
    给定一个平面的原点和法向量,找到将该平面移动到与XY平面共面的变换矩阵

    :param origin: (3,) float,空间中的点
    :param normal: (3,) float,平面的法向量
    :return: transform: (4,4) float,变换矩阵
    '''
    transform = align_vectors(normal, [0, 0, 1])
    transform[0:3, 3] = -np.dot(transform, np.append(origin, 1))[0:3]
    return transform


def transform_around(matrix, point):
    '''
    给定一个变换矩阵,围绕空间中的一个点应用其旋转分量

    :param matrix: (4,4) float,变换矩阵
    :param point: (3,) float,空间中的点
    :return: result: (4,4) 变换矩阵
    '''
    point = np.array(point)
    translate = np.eye(4)
    translate[0:3, 3] = -point
    result = np.dot(matrix, translate)
    translate[0:3, 3] = point
    result = np.dot(translate, result)
    return result


def align_vectors(vector_start, vector_end, return_angle=False):
    '''
    返回一个4x4的变换矩阵

    该矩阵将从vector_start (3,)旋转到vector_end (3,),例如:
    vector_end == np.dot(T, np.append(vector_start, 1))[0:3]
    '''
    # 以下代码由weiwei于07212017添加 用于修正相同向量和反向向量的问题
    if np.array_equal(vector_start, vector_end):
        T = np.eye(4)
        angle = 0.0
        if return_angle:
            return T, angle
        return T

    # 胡老师代码
    if np.array_equal(-vector_start, vector_end):
        a = vector_start[0]
        b = vector_start[1]
        c = vector_start[2]
        rot_ax = unitize(np.array([b - c, -a + c, a - b]))
        T = rotation_matrix(np.pi, rot_ax)
        if return_angle:
            return T, np.pi
        return T

    # # 陈老师代码
    # if np.array_equal(-vector_start, vector_end):
    #     T = np.eye(4)
    #     T[:3, 2] *= -1.0
    #     T[:3, 1] *= -1.0
    #     angle = np.pi
    #     if return_angle:
    #         return T, angle
    #     return T

    vector_start = unitize(vector_start)
    vector_end = unitize(vector_end)
    cross = np.cross(vector_start, vector_end)
    # 我们将范数剪辑到1,否则浮点数问题 可能导致arcsin出错
    norm = np.clip(np.linalg.norm(cross), -1.0, 1.0)
    direction = np.sign(np.dot(vector_start, vector_end))
    if norm < tol.zero:
        # 如果范数为零,向量相同 不需要旋转
        T = np.eye(4)
        T[0:3] *= direction
    else:
        angle = np.arcsin(norm)
        if direction < 0:
            angle = np.pi - angle
        T = rotation_matrix(angle, cross)

    check = np.dot(T[:3, :3], vector_start) - vector_end
    if not np.allclose(check, 0.0):
        raise ValueError('Vectors unaligned!')

    if return_angle:
        return T, angle
    return T


def faces_to_edges(faces, return_index=False):
    '''
    给定一个面列表 (n,3),返回一个边列表 (n*3,2)

    :param faces: (n,3) int,面顶点索引列表
    :param return_index: bool,是否返回面索引
    :return: edges: (n*3,2) int,边顶点索引列表
    '''
    faces = np.asanyarray(faces)
    edges = np.column_stack((faces[:, (0, 1)],
                             faces[:, (1, 2)],
                             faces[:, (2, 0)])).reshape(-1, 2)
    if return_index:
        face_index = np.tile(np.arange(len(faces)), (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges


def triangulate_quads(quads):
    '''
    给定一组四边形面,将它们返回为三角形面

    :param quads: (n,4) int,四边形面顶点索引列表
    :return: faces: (n*2,3) int,三角形面顶点索引列表
    '''
    quads = np.array(quads)
    faces = np.vstack((quads[:, [0, 1, 2]],
                       quads[:, [2, 3, 0]]))
    return faces


def mean_vertex_normals(vertex_count, faces, face_normals, **kwargs):
    '''
    从包含该顶点的面的平均值中找到顶点法线

    :param vertex_count: int,顶点数量
    :param faces: (n,3) int,面顶点索引列表
    :param face_normals: (n,3) float,每个面的法向量
    :return: vertex_normals: (vertex_count, 3) float,每个顶点的法线
    '''

    def summed_sparse():
        # 使用稀疏矩阵来确定每个顶点的总法线
        if 'sparse' in kwargs:
            sparse = kwargs['sparse']
        else:
            sparse = index_sparse(vertex_count, faces)
        summed = sparse.dot(face_normals)
        log.debug('使用稀疏矩阵生成顶点法线')
        return summed

    def summed_loop():
        # 遍历每个面,在测试中比使用稀疏矩阵慢约50倍
        summed = np.zeros((vertex_count, 3))
        for face, normal in zip(faces, face_normals):
            summed[face] += normal
        return summed

    try:
        summed = summed_sparse()
    except:
        log.warning('无法生成稀疏矩阵！回退到循环方法!', exc_info=True)
        summed = summed_loop()
    unit_normals, valid = unitize(summed, check_valid=True)
    vertex_normals = np.zeros((vertex_count, 3), dtype=np.float64)
    vertex_normals[valid] = unit_normals
    return vertex_normals


def index_sparse(column_count, indices):
    '''
    返回一个稀疏矩阵,表示哪些顶点包含在哪些面中

    :param column_count: int,列数(顶点数量)
    :param indices: (n,3) int,面顶点索引列表
    :return: sparse: scipy.sparse.coo_matrix,形状为 (column_count, len(faces)),数据类型为布尔型

    Example
     ----------
    In [1]: sparse = faces_sparse(len(mesh.vertices), mesh.faces)

    In [2]: sparse.shape
    Out[2]: (12, 20)

    In [3]: mesh.faces.shape
    Out[3]: (20, 3)

    In [4]: mesh.vertices.shape
    Out[4]: (12, 3)

    In [5]: dense = sparse.toarray().astype(int)

    In [6]: dense
    Out[6]:
    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]])

    In [7]: dense.sum(axis=0)
    Out[7]: array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    '''
    indices = np.asanyarray(indices)
    column_count = int(column_count)

    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape((-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (column_count, len(indices))
    data = np.ones(len(col), dtype=bool)
    sparse = coo_matrix((data, (row, col)),
                        shape=shape,
                        dtype=bool)
    return sparse


def medial_axis(samples, contains):
    '''
    给定边界上的一组样本,基于 Voronoi图和一个可以评估点是否在封闭几何体内或外的包含函数,找到近似的中轴

    :param samples: (n,d) float,几何体边界上的点集
    :param contains: function,接受 (m,d) 点并返回 (m) 布尔数组的函数,判断点是否在几何体内
    :return: lines: (n,2,2) float,线段集
    '''

    from scipy.spatial import Voronoi
    from .path.io.load import load_path

    # 创建Voronoi图,在垂直堆叠点之后 将一个序列deque为一个干净的(m,2)数组
    voronoi = Voronoi(samples)
    # 哪些Voronoi顶点包含在原始多边形内
    contained = contains(voronoi.vertices)
    # -1的ridge顶点在外部,确保它们为False
    contained = np.append(contained, False)
    inside = [i for i in voronoi.ridge_vertices if contained[i].all()]
    line_indices = np.vstack([stack_lines(i) for i in inside if len(i) >= 2])
    lines = voronoi.vertices[line_indices]
    return load_path(lines)


def rotation_2D_to_3D(matrix_2D):
    '''
    给定一个二维齐次旋转矩阵,将其转换为绕Z轴旋转的三维旋转矩阵

    :param matrix_2D: (3,3) float,齐次二维旋转矩阵
    :return: matrix_3D: (4,4) float,齐次三维旋转矩阵
    '''
    matrix_2D = np.asanyarray(matrix_2D)
    if matrix_2D.shape != (3, 3):
        raise ValueError('需要齐次二维变换矩阵!')
    matrix_3D = np.eye(4)
    # 平移
    matrix_3D[0:2, 3] = matrix_2D[0:2, 2]
    # 从二维到绕Z轴的旋转
    matrix_3D[0:2, 0:2] = matrix_2D[0:2, 0:2]
    return matrix_3D
