import numpy as np
import networkx as nx
from collections import deque
from .geometry import faces_to_edges
from .grouping import group_rows
from .triangles import normals
from .util import is_sequence
from .constants import log, tol


def fix_face_winding(mesh):
    '''
    遍历并原地修改网格面以确保绕线一致
    即相邻面的边缘方向相反
    '''
    if mesh.is_winding_consistent:
        log.info('网孔有一致绕组,退出绕组修复')
        return
    # 我们创建人脸邻接图:
    # g中的每个节点都是mesh.faces的索引
    # g中的每条边表示连接的两个面
    graph_all = nx.from_edgelist(mesh.face_adjacency)
    flipped = 0
    faces = mesh.faces.view(np.ndarray).copy()
    # 我们将使用BFS遍历图,因此我们必须开始遍历每个连通组件
    for graph in (graph_all.subgraph(c) for c in nx.connected_components(graph_all)):
        start = next(iter(graph.nodes()))
        # 我们遍历图中的每一对面我们修改网格.面和网格.Face_normals就位
        for face_pair in nx.bfs_edges(graph, start):
            # 对于每对面,我们将其转换为边,找出两个面共享的边,然后看看这些边是否顺序颠倒,就像你在一个构造良好的网格中所期望的那样
            face_pair = np.ravel(face_pair)
            pair = faces[face_pair]
            edges = faces_to_edges(pair)
            overlap = group_rows(np.sort(edges, axis=1), require_count=2)
            if len(overlap) == 0:
                # 只发生在非水密的网格上
                continue
            edge_pair = edges[overlap[0]]
            if edge_pair[0][0] == edge_pair[1][0]:
                # 如果边不反转,反转其中一个面的顺序
                flipped += 1
                faces[face_pair[1]] = faces[face_pair[1]][::-1]
    if flipped > 0:
        mesh.faces = faces
    log.info('Flipped %d/%d edges', flipped, len(mesh.faces) * 3)


def fix_normals_direction(mesh):
    '''
    检查网格的法线是否指向实体外部,使用射线测试

    如果网格不是密封的,这将没有意义
    '''
    # 根据右手法则重新生成面法向量
    mesh.face_normals = None
    # 测试射线的方向
    direction = mesh.face_normals[0]
    # test point
    origin = mesh.triangles[0].mean(axis=0)
    origin += direction * tol.merge
    flipped = mesh.contains([origin])[0]
    if flipped:
        log.debug('翻转面法线并缠绕')
        # 反转面法线
        mesh.face_normals *= -1.0
        # 由于法线被重新生成,这意味着旋回是向后的,如果缠绕是不连贯的,这不会解决任何问题
        mesh.faces = np.fliplr(mesh.faces)


def fix_normals(mesh):
    '''
    原地修复网格面的绕线和法线方向

    这实际上只对密封网格有意义,但也会以统一的方式
    定向非密封面片的所有面和绕线.
    '''
    fix_face_winding(mesh)
    fix_normals_direction(mesh)


def broken_faces(mesh, color=None):
    '''
    返回破坏网格密封状态的面的索引
    如果设置了颜色,则更改破损面的颜色
    '''
    adjacency = nx.from_edgelist(mesh.face_adjacency)
    broken = [k for k, v in adjacency.degree().iteritems() if v != 3]
    broken = np.array(broken)
    if color is not None:
        if not is_sequence(color): color = [255, 0, 0]
        mesh.visual.face_colors[broken] = color
    return broken


def fill_holes(mesh):
    '''
    通过添加新三角形填充三角网格上的单个三角形孔,新三角形将具有正确的绕线和法线,如果存在面颜色
    则将最后一个面的颜色分配给新三角形

    :param mesh: Trimesh 对象
    '''

    def hole_to_faces(hole):
        '''
        给定表示孔的顶点索引循环,将其转换为三角形面
        如果无法做到这一点,则返回 None

        :param hole: 有序的顶点索引循环

        :return: (n, 3) 新面
                 (m, 3) 新顶点
        '''
        hole = np.asanyarray(hole)
        # 洞只是一个缺失的三角形的情况
        if len(hole) == 3:
            return [hole], []
        # 这个洞是一个四边形,我们用两个三角形填充它
        if len(hole) == 4:
            face_A = hole[[0, 1, 2]]
            face_B = hole[[2, 3, 0]]
            return [face_A, face_B], []
        return [], []

    if len(mesh.faces) < 3:
        return False
    edges = mesh.edges
    edges_sorted = np.sort(edges, axis=1)
    # 我们知道在一个水密网格中,每条边都会被包含两次
    # 因此,每个只出现一次的边缘都是孔边界的一部分
    boundary_groups = group_rows(edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        watertight = len(boundary_groups) == 0
        return watertight
    boundary_edges = edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]
    # 我们创建一个边界边的图,并找到环
    graph = nx.from_edgelist(np.column_stack((boundary_edges, index_as_dict)))
    cycles = np.array(nx.cycle_basis(graph))
    new_faces = deque()
    new_vertex = deque()
    for hole in cycles:
        # 将空洞转换为三角形和新顶点的多边形
        faces, vertex = hole_to_faces(hole=hole)
        if len(faces) == 0:
            continue
        # 重新网格化会以负索引返回新顶点,所以请更改它们到绝对索引,不会被后面的追加搞砸
        faces = np.array(faces)
        faces[faces < 0] += len(new_vertex) + len(mesh.vertices) + len(vertex)
        new_vertex.extend(vertex)
        new_faces.extend(faces)
    new_faces = np.array(new_faces)
    new_vertex = np.array(new_vertex)

    if len(new_faces) == 0:
        # 没有添加新面孔,所以没有进一步做的网格不是水密的,因为边界组存在,但我们没有添加任何新面孔来填充它们
        return False
    for face_index, face in enumerate(new_faces):
        # 我们将来自新人脸的边缘与源网格的边界边缘
        edge_test = face[0:2]
        edge_boundary = edges[graph.get_edge_data(*edge_test)['index']]
        # 在一个构造良好的网格中,缠绕是这样的,即相邻的三角形
        # 相互反转的边.这里我们检查以确保
        # 边是反转的,如果不是,我们简单地反转面
        reversed = edge_test[0] == edge_boundary[1]
        if not reversed:
            new_faces[face_index] = face[::-1]
    if len(new_vertex) != 0:
        new_vertices = np.vstack((mesh.vertices, new_vertex))
    else:
        new_vertices = mesh.vertices
    # 由于绕线正确,我们可以得到一致的法线只需要在面边缘做叉乘
    mesh._cache.clear(exclude=['face_normals'])
    new_normals, valid = normals(new_vertices[new_faces])
    mesh.face_normals = np.vstack((mesh.face_normals, new_normals))
    mesh.faces = np.vstack((mesh._data['faces'], new_faces[valid]))
    mesh.vertices = new_vertices
    mesh._cache.id_set()
    # 这通常是一个三角形的两个顶点刚好 over tol.合并分开,但正常的计算是错误的
    # 这些问题可以通过合并这里的顶点来解决:
    # 如果无效,则.all():
    # 打印有效内容
    if mesh.visual.defined and mesh.visual._set['face']:
        # 如果存在face颜色,则将最后一个颜色分配给新face
        # 请注意,这有点奶酪,但它非常便宜,如果网格是单一颜色,这是正确的选择
        stored = mesh.visual._data['face_colors']
        color_shape = np.shape(stored)
        if len(color_shape) == 2:
            new_colors = np.tile(stored[-1], (np.sum(valid), 1))
            new_colors = np.vstack((stored,
                                    new_colors))
            mesh.visual.face_colors = new_colors
    log.debug('用 %i 三角形填充网格', np.sum(valid))
    return mesh.is_watertight
