import numpy as np
import copy
import random
import networkx as nx
import math

from collections import deque

from .constants import log, tol
from .grouping import group, group_rows, boolean_rows
from .geometry import faces_to_edges
from .util import diagonal_dot

try:
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components

    _has_gt = True
except:
    _has_gt = False
    log.warning('graph-tool不可用,某些操作会慢得多')


def face_adjacency(faces, return_edges=False):
    '''
    返回一个面索引列表,其中每对面共享一个边,使它们相邻

    :param faces: (n, d) int,面集合,通过索引引用顶点
    :param return_edges: bool,返回相邻面共享的边
    :return: adjacency: (m,2) int,相邻面的索引
             如果 return_edges 为 True:
             edges: (m,2) int,构成相邻面共享边的顶点索引

    示例
    ----------
    这对于很多事情都很有用,比如找到连通组件:

    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency)
    groups = nx.connected_components(graph_connected)
    '''
    # 首先生成当前面的边列表 还返回边所属面的索引
    edges, edge_face_index = faces_to_edges(faces, return_index=True)
    edges.sort(axis=1)
    # 这将返回重复边的索引
    # 在构造良好的网格中,每条边出现两次
    # 所以对于 edge_idx 中的每一行,edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # 在这次调用中,我们丢弃不出现两次的边
    edge_groups = group_rows(edges, require_count=2)
    if len(edge_groups) == 0:
        log.error('未检测到相邻面！您是否合并了顶点？')

    # 所有相邻面的对
    # 所以对于 face_idx 中的每一行,self.faces[face_idx[*][0]] 和 self.faces[face_idx[*][1]] 将共享一个边
    face_adjacency = edge_face_index[edge_groups]
    if return_edges:
        face_adjacency_edges = edges[edge_groups[:, 0]]
        return face_adjacency, face_adjacency_edges
    return face_adjacency


def adjacency_angle(mesh, angle, direction=np.less, return_edges=False):
    '''
    仅当网格的面小于指定角度时,返回相邻面

    :param mesh: Trimesh 对象
    :param angle: float,默认以弧度为单位的角度,角度大于此值的面将被视为不相邻
    :param direction: function,用于测试面角度与角度参数,默认设置为 np.less
    :param return_edges: bool,是否返回与相邻性相关的边
    :return: adjacency: (n,2) int 网格中的面索引列表
             如果 return_edges 为 True:
             edges: (n,2) int 网格中的顶点索引列表(边)
    '''
    # 尽可能使用缓存的相邻性 (n,2)
    adjacency = mesh.face_adjacency
    # 相邻面的法向量 (n, 2, 3)
    normals = mesh.face_normals[adjacency]
    # 法向量的点积 (n)
    dots = diagonal_dot(normals[:, 0], normals[:, 1])
    # 剪裁浮点误差
    dots = np.clip(dots, -1.0, 1.0)
    adj_ok = direction(np.abs(np.arccos(dots)), angle)
    # 结果是 (m,2)
    new_adjacency = adjacency[adj_ok]
    if return_edges:
        edges = mesh.face_adjacency_edges[adj_ok]
        return new_adjacency, edges
    return new_adjacency


def shared_edges(faces_a, faces_b):
    '''
    给定两组面,找到两组中都存在的边

    :param faces_a: (n,3) int,面集合
    :param faces_b: (m,3) int,面集合
    :return: shared: (p, 2) int,边集合
    '''
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = boolean_rows(e_a, e_b, operation=set.intersection)
    return shared


def connected_edges(G, nodes):
    '''
    给定图 G 和节点列表,返回与节点连接的边列表

    :param G: 图对象
    :param nodes: 节点列表
    :return: edges: 与节点连接的边列表
    '''
    nodes_in_G = deque()
    for node in nodes:
        if not G.has_node(node):
            continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges


def facets(mesh):
    '''
    找到平行相邻面的列表

    :param mesh: Trimesh
    :return: facets: 平行相邻面的面索引组列表(在 mesh.faces 中)
    '''

    def facets_nx():
        graph_parallel = nx.from_edgelist(face_idx[parallel])
        facets_idx = np.array([list(i) for i in nx.connected_components(graph_parallel)])
        # weiwei 注释  还应该返回单个三角形
        facets_idx_extra = copy.deepcopy(facets_idx.tolist())
        for item in range(mesh.faces.shape[0]):
            if item not in [i for subitem in facets_idx.tolist() for i in subitem]:
                facets_idx_extra.append([item])
        return np.array(facets_idx_extra)
        # return facets_idx

    def facets_gt():
        # 使用 graph-tool 库创建一个图
        graph_parallel = GTGraph()
        # 添加平行面的边
        graph_parallel.add_edge_list(face_idx[parallel])
        # 找到连通组件
        connected = label_components(graph_parallel, directed=False)[0].a
        # 将连通组件分组
        facets_idx = group(connected, min_len=2)
        return facets_idx

    # (n,2) 相邻面索引列表
    face_idx = mesh.face_adjacency

    # 测试相邻面的角度
    normal_pairs = mesh.face_normals[tuple([face_idx])]
    normal_dot = (np.sum(normal_pairs[:, 0, :] * normal_pairs[:, 1, :], axis=1) - 1) ** 2

    # 如果法向量实际上相等,则它们是平行的,具有很高的置信度
    parallel = normal_dot < tol.zero
    non_parallel = np.logical_not(parallel)

    # 说两个面*不*平行容易出错
    # 因此我们添加一个半径检查,计算面心之间的距离
    # 并将其除以法向量的点积
    # 这意味着大面之间的小角度将有一个大半径,我们可以轻松过滤掉
    # 如果不这样做,微小面的浮点误差可能会将法向量推过纯角度阈值,即使面上的实际偏差极小
    center = mesh.triangles.mean(axis=1)
    center_sq = np.sum(np.diff(center[face_idx], axis=1).reshape((-1, 3)) ** 2, axis=1)
    radius_sq = center_sq[non_parallel] / normal_dot[non_parallel]
    parallel[non_parallel] = radius_sq > tol.facet_rsq

    # weiwei注释,始终使用 graph-nx
    # graph-tool 比 networkx 快约6倍,但更难安装
    # if _has_gt: return facets_gt()
    # else:       return facets_nx()

    return facets_nx()


def facets_over_segmentation(mesh, faceangle=.9, segangle=.9):
    """
    使用网格过分割计算簇

    :param mesh: Trimesh 对象
    :param faceangle: 两个相邻面被视为共面的角度
    :param segangle: 两个相邻分割被视为共面的角度
    :return: 与 facets 相同的结果

    author: weiwei
    date: 20161116cancun, 20210119osaka
    """
    def __expand_adj(mesh, face_id, reference_normal, adjacent_face_list, face_angle=.9):
        """
        找到面的邻接
        新添加面的法向量应与参考法向量一致
        这是一个迭代函数

        :param mesh: Trimesh 对象
        :param face_id: 要扩展的面的索引
        :param reference_normal: 参考面的法向量
        :param adjacent_face_list: 面的邻接列表
        :param face_angle: 被视为共面的相邻面角度
        :return: None 或 face_ids 列表

        author: weiwei
        date: 20161213, 20210119
        """
        curvature = 0
        face_center = np.mean(mesh.vertices[mesh.faces[face_id]], axis=1)
        bool_list0 = np.asarray(adjacent_face_list[:, 0] == face_id)
        bool_list1 = np.asarray(adjacent_face_list[:, 1] == face_id)
        xadjid0 = adjacent_face_list[bool_list1][:, 0]
        xadjid1 = adjacent_face_list[bool_list0][:, 1]
        xadjid = np.append(xadjid0, xadjid1)
        returnlist = []
        faceadjx = adjacent_face_list[np.logical_not(np.logical_or(bool_list0, bool_list1))]
        for j in xadjid:
            newnormal = mesh.face_normals[j]
            dotnorm = np.dot(reference_normal, newnormal)
            if dotnorm > face_angle:
                newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                d = np.linalg.norm(newfacecenter - face_center)
                if dotnorm > 1.0:
                    dotnorm = 1.0
                tmp_curvature = math.acos(dotnorm) / d
                if tmp_curvature > curvature:
                    curvature = tmp_curvature
                returnlist.append(j)
        finalreturnlist = [face_id]
        while returnlist:
            finalreturnlist.extend(returnlist)
            finalreturnlist = list(set(finalreturnlist))
            newreturnlist = []
            for id, j in enumerate(returnlist):
                bool_list0 = np.asarray(faceadjx[:, 0] == j)
                bool_list1 = np.asarray(faceadjx[:, 1] == j)
                xadjid0 = faceadjx[bool_list1][:, 0]
                xadjid1 = faceadjx[bool_list0][:, 1]
                xadjid = np.append(xadjid0, xadjid1)
                faceadjx = faceadjx[np.logical_not(np.logical_or(bool_list0, bool_list1))]
                for k in xadjid:
                    newnormal = mesh.face_normals[k]
                    dotnorm = np.dot(reference_normal, newnormal)
                    if dotnorm > face_angle:
                        newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                        d = np.linalg.norm(newfacecenter - face_center)
                        if dotnorm > 1.0:
                            dotnorm = 1.0
                        tmp_curvature = math.acos(dotnorm) / d
                        if tmp_curvature > curvature:
                            curvature = tmp_curvature
                        newreturnlist.append(k)
            returnlist = list(set(newreturnlist))
        return finalreturnlist, curvature

    # plot using panda3d
    # import trimesh.visual as visual

    # from panda3d.core import GeomNode, NodePath, Vec4
    # import pandaplotutils.pandactrl as pandactrl
    # import pandaplotutils.pandageom as pandageom
    # base = pandactrl.World(camp=[700, 300, 700], lookatp=[0, 0, 0])

    # the approach using random start
    # faceadj  = mesh.face_adjacency
    # removelist = []
    # faceids = list(range(len(mesh.faces)))
    # random.shuffle(faceids)
    # for i in faceids:
    #     if i not in removelist:
    #         print i, len(mesh.faces)
    #         rndcolor = visual.color_to_float(visual.random_color())
    #         adjidlist = __expand_adj(mesh, i, mesh.face_normals[i], faceadj)
    #         removelist.extend(adjidlist)
    #         for j in adjidlist:
    #             vertxid = mesh.faces[j, :]
    #             vert0 = mesh.vertices[vertxid[0]]
    #             vert1 = mesh.vertices[vertxid[1]]
    #             vert2 = mesh.vertices[vertxid[2]]
    #             verts = np.array([vert0, vert1, vert2])
    #             normals = mesh.face_normals[j].reshape(1,3)
    #             triangles = np.array([[0, 1, 2]])
    #             geom = pandageom.packpandageom(verts, normals, triangles)
    #             node = GeomNode('piece')
    #             node.addGeom(geom)
    #             star = NodePath('piece')
    #             star.attachNewNode(node)
    #             star.setColor(Vec4(rndcolor[0], rndcolor[1], rndcolor[2], rndcolor[3]))
    #             star.setTwoSided(True)
    #             star.reparentTo(base.render)
    # base.run()

    # 使用大的法向量差异的方法
    faceadj = mesh.face_adjacency
    faceids = list(range(len(mesh.faces)))
    random.shuffle(faceids)
    knownfacetnormals = np.array([])
    knownfacets = []
    knowncurvature = []
    for i in faceids:
        if knownfacetnormals.size:
            potentialfacetsidx = np.where(np.dot(knownfacetnormals, mesh.face_normals[i]) > segangle)[0]
            # 对于平行面
            potentialfaceids = []
            if potentialfacetsidx.size:
                for pfi in potentialfacetsidx:
                    potentialfaceids.extend(knownfacets[pfi])
                if i in potentialfaceids:
                    continue

        # 扩展邻接面,计算曲率
        # rndcolor = visual.color_to_float(visual.random_color())
        adjidlist, curvature = __expand_adj(mesh, i, mesh.face_normals[i], faceadj, faceangle)
        facetnormal = np.sum(mesh.face_normals[adjidlist], axis=0)
        facetnormal = facetnormal / np.linalg.norm(facetnormal)  # 归一化法向量
        if knownfacetnormals.size:
            knownfacetnormals = np.vstack((knownfacetnormals, facetnormal))
        else:
            knownfacetnormals = np.hstack((knownfacetnormals, facetnormal))
        knownfacets.append(adjidlist)
        knowncurvature.append(curvature)
    return [np.array(knownfacets), np.array(knownfacetnormals), np.array(knowncurvature)]

    # plot using panda3d
    #     for j in adjidlist:
    #         vertxid = mesh.faces[j, :]
    #         vert0 = mesh.vertices[vertxid[0]]+.012*i*facetnormal
    #         vert1 = mesh.vertices[vertxid[1]]+.012*i*facetnormal
    #         vert2 = mesh.vertices[vertxid[2]]+.012*i*facetnormal
    #         verts = np.array([vert0, vert1, vert2])
    #         normals = mesh.face_normals[j].reshape(1,3)
    #         triangles = np.array([[0, 1, 2]])
    #         geom = pandageom.packpandageom(verts, normals, triangles)
    #         node = GeomNode('piece')
    #         node.addGeom(geom)
    #         star = NodePath('piece')
    #         star.attachNewNode(node)
    #         star.setColor(Vec4(rndcolor[0], rndcolor[1], rndcolor[2], rndcolor[3]))
    #         star.setTwoSided(True)
    #         star.reparentTo(base.render)
    # base.run()

    # for i in range(1122,1123):
    #     rndcolor = visual.color_to_float(visual.DEFAULT_COLOR)
    #     vertxid = mesh.faces[i, :]
    #     vert0 = mesh.vertices[vertxid[0]]
    #     vert1 = mesh.vertices[vertxid[1]]
    #     vert2 = mesh.vertices[vertxid[2]]
    #     verts = [[vert0, vert1, vert2]]
    #     tri = Poly3DCollection(verts)
    #     tri.set_color([rndcolor])
    #     ax.add_collection3d(tri)
    # plt.show()


def facets_noover(mesh, faceangle=.9):
    """
    使用网格分割计算面簇."noover"表示不考虑重叠

    :param mesh: Trimesh 对象,表示网格
    :param faceangle: float,相邻面之间被视为共面的角度
    :return: 与 facets 相同的结果

    author: weiwei
    date: 20161116, cancun
    """

    def __expand_adj(mesh, faceid, refnormal, faceadj, faceangle, knownfacets):
        """
        找到一个面的邻接面,新添加面的法向量应与参考法向量一致,这是一个迭代函数

        :param mesh: Trimesh 对象
        :param faceid: int,要扩展的面的索引
        :param refnormal: array,参考面的法向量
        :param faceadj: array,面的邻接列表
        :param faceangle: float,相邻面之间被视为共面的角度
        :return: None 或一个面索引列表

        author: weiwei
        date: 20161213, update faceangle
        """
        knownidlist = [i for facet in knownfacets for i in facet]
        curvature = 0
        facecenter = np.mean(mesh.vertices[mesh.faces[faceid]], axis=1)

        boollist0 = np.asarray(faceadj[:, 0] == faceid)
        boollist1 = np.asarray(faceadj[:, 1] == faceid)
        xadjid0 = faceadj[boollist1][:, 0]
        xadjid1 = faceadj[boollist0][:, 1]
        xadjid = np.append(xadjid0, xadjid1)
        returnlist = []
        faceadjx = faceadj[np.logical_not(np.logical_or(boollist0, boollist1))]
        for j in xadjid:
            if j in knownidlist:
                continue
            newnormal = mesh.face_normals[j]
            dotnorm = np.dot(refnormal, newnormal)
            if dotnorm > faceangle:
                newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                d = np.linalg.norm(newfacecenter - facecenter)
                if dotnorm > 1.0:
                    dotnorm = 1.0
                tempcurvature = math.acos(dotnorm) / d
                if tempcurvature > curvature:
                    curvature = tempcurvature
                returnlist.append(j)
        finalreturnlist = [faceid]
        while returnlist:
            finalreturnlist.extend(returnlist)
            finalreturnlist = list(set(finalreturnlist))
            newreturnlist = []
            for id, j in enumerate(returnlist):
                if j in knownidlist:
                    continue
                boollist0 = np.asarray(faceadjx[:, 0] == j)
                boollist1 = np.asarray(faceadjx[:, 1] == j)
                xadjid0 = faceadjx[boollist1][:, 0]
                xadjid1 = faceadjx[boollist0][:, 1]
                xadjid = np.append(xadjid0, xadjid1)
                faceadjx = faceadjx[np.logical_not(np.logical_or(boollist0, boollist1))]
                for k in xadjid:
                    newnormal = mesh.face_normals[k]
                    dotnorm = np.dot(refnormal, newnormal)
                    if dotnorm > faceangle:
                        newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                        d = np.linalg.norm(newfacecenter - facecenter)
                        if dotnorm > 1.0:
                            dotnorm = 1.0
                        tempcurvature = math.acos(dotnorm) / d
                        if tempcurvature > curvature:
                            curvature = tempcurvature
                        newreturnlist.append(k)
            returnlist = list(set(newreturnlist))
        return finalreturnlist, curvature

    # 使用大的法向量差异的方法
    knownfacetnormals = np.array([])
    knownfacets = []
    knowncurvature = []
    faceadj = mesh.face_adjacency
    faceids = list(range(len(mesh.faces)))
    while True:
        random.shuffle(faceids)
        i = faceids[0]
        adjidlist, curvature = __expand_adj(mesh, i, mesh.face_normals[i], faceadj, faceangle, knownfacets)
        facetnormal = np.sum(mesh.face_normals[adjidlist], axis=0)
        facetnormal = facetnormal / np.linalg.norm(facetnormal)
        if knownfacetnormals.size:
            knownfacetnormals = np.vstack((knownfacetnormals, facetnormal))
        else:
            knownfacetnormals = np.hstack((knownfacetnormals, facetnormal))
        knownfacets.append(adjidlist)
        knowncurvature.append(curvature)
        faceids = list(set(faceids) - set(adjidlist))
        if len(faceids) == 0:
            break
    # 胡老师
    # return [np.array(knownfacets, dtype="objects"), np.array(knownfacetnormals, dtype="objects"),
    #  np.array(knowncurvature, dtype="objects")]

    # 陈老师
    return [np.array(knownfacets), np.array(knownfacetnormals), np.array(knowncurvature)]


def split(mesh, only_watertight=True, adjacency=None):
    '''
    给定一个网格,将其根据面连接性分割成多个网格.如果 only_watertight 为真,则只返回每个面恰好有三个相邻面的网格

    :param mesh: Trimesh 对象
    :param only_watertight: bool,如果为真,只返回水密组件
    :param adjacency: (n,2) 面邻接列表,用于覆盖自动计算的邻接
    :return: list of Trimesh 对象
    '''

    def split_nx():
        # 使用 NetworkX 分割网格
        adjacency_graph = nx.from_edgelist(adjacency)
        components = nx.connected_components(adjacency_graph)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    def split_gt():
        # 使用 graph-tool 分割网格
        g = GTGraph()
        g.add_edge_list(adjacency)
        component_labels = label_components(g, directed=False)[0].a
        components = group(component_labels)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    if adjacency is None:
        adjacency = mesh.face_adjacency

    if _has_gt:
        return split_gt()
    else:
        return split_nx()


def smoothed(mesh, angle):
    '''
    返回一个非水密版本的网格,该网格在平滑着色下渲染效果良好

    :param mesh: Trimesh 对象
    :param angle: float,弧度,相邻面法向量小于此角度的将被平滑
    :return: Trimesh 对象
    '''
    adjacency = adjacency_angle(mesh, angle)
    graph = nx.from_edgelist(adjacency)
    graph.add_nodes_from(np.arange(len(mesh.faces)))
    smooth = mesh.submesh(nx.connected_components(graph),
                          only_watertight=False,
                          append=True)
    return smooth


def is_watertight(edges, return_winding=False):
    '''
    检查每条边是否被两个面包含

    :param edges: (n,2) int,顶点索引集合
    :return: boolean,是否每条边都被两个面包含
    '''
    edges_sorted = np.sort(edges, axis=1)
    groups = group_rows(edges_sorted, require_count=2)
    watertight = (len(groups) * 2) == len(edges)
    if return_winding:
        opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
        reversed = np.equal(*opposing).all()
        return watertight, reversed
    return watertight
