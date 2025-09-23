import numpy as np
from .grouping import unique_rows


def subdivide(mesh, face_index=None):
    '''
    将网格细分为更小的三角形

    :param mesh: Trimesh 对象
    :param face_index: 要细分的面
                       如果为 None:网格的所有面都将被细分
                       如果为 (n,) 的整型数组:只有指定的面将被细分
                       请注意,在这种情况下,网格通常不再是流形,因为在中点上的附加顶点不会被指定面的相邻面使用
                       并且需要额外的后处理步骤以使结果网格密封
    '''
    if face_index is None:
        face_index = np.arange(len(mesh.faces))
    else:
        face_index = np.asanyarray(face_index, dtype=np.int64)
    # 三角形中的(c, 3, 3)浮点数点集
    faces = mesh.faces[face_index]
    # 每个三角形边的3个中点v堆叠成(3*c, 3)浮点数
    triangles = mesh.triangles[face_index]
    # 每个三角形边的3个中点v堆叠成(3*c, 3)浮点数
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                               [1, 2],
                                                               [2, 0]]])
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    # 对于相邻的面,我们将生成相同的中点 两次,所以我们通过找到唯一顶点来处理它
    unique, inverse = unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(mesh.vertices)
    # 新面孔,正确的绕线
    f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                         mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                         mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                         mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
    # 为每张旧面孔添加3张新面孔
    new_faces = np.vstack((mesh.faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]
    mesh.vertices = np.vstack((mesh.vertices, mid))
    mesh.faces = new_faces
