# 在 panda3d 和 trimesh 之间转换数据的适配器文件

from pyglet.math import Vec4
import basis.trimesh as trm
import numpy as np
from panda3d.core import Geom, GeomNode, GeomPoints, GeomTriangles, GeomVertexWriter
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexArrayFormat, InternalName
from panda3d.core import GeomEnums
from panda3d.core import NodePath, Vec3, Mat3, Mat4, LQuaternion
from typing import List


# data manipulation
def gen_colorarray(ncolors=1, alpha=1, nonrandcolor=None) -> List[List[float]]:
    """
    生成一个颜色数组

    如果ncolors = 1,返回一个包含4个元素的列表(RGBA)

    :param ncolors: 生成颜色的数量
    :param alpha: 透明度,默认为1(完全不透明)
    :param nonrandcolor: 如果提供,返回此固定颜色,否则生成随机颜色
    :return: 颜色数组,每个颜色为一个包含RGB和alpha的4元素列表

    作者: weiwei
    日期: 20161130大阪
    """
    if ncolors == 1:
        if nonrandcolor:
            return [nonrandcolor[0], nonrandcolor[1], nonrandcolor[2], alpha]
        else:
            # 否则生成一个随机颜色
            return [np.random.random(), np.random.random(), np.random.random(), alpha]
    colorarray = []
    # 如果需要生成多个颜色
    for i in range(ncolors):
        # 如果提供了固定颜色,则返回固定颜色
        if nonrandcolor:
            colorarray.append([nonrandcolor[0], nonrandcolor[1], nonrandcolor[2], alpha])
        else:
            # 否则生成一个随机颜色
            colorarray.append([np.random.random(), np.random.random(), np.random.random(), alpha])
    return colorarray


def npmat3_to_pdmat3(npmat3):
    """
    将一个3x3的NumPy数组转换为Panda3D中的LMatrix3f对象

    :param npmat3: 一个3x3的NumPy ndarray
    :return: 转换后的LMatrix3f对象,Panda3D中的3x3矩阵

    作者: weiwei
    日期: 20161107
    """
    # 从NumPy数组提取每个元素,并使用这些元素创建一个Panda3D的3x3矩阵
    return Mat3(npmat3[0, 0], npmat3[1, 0], npmat3[2, 0], \
                npmat3[0, 1], npmat3[1, 1], npmat3[2, 1], \
                npmat3[0, 2], npmat3[1, 2], npmat3[2, 2])


def pdmat3_to_npmat3(pdmat3):
    """
    将Panda3D的mat3矩阵转换为NumPy的二维数组

    :param pdmat3: Panda3D的mat3对象
    :return: 对应的NumPy二维数组,表示相同的3x3矩阵

    作者: weiwei
    日期: 20161216札幌
    """
    # 从Panda3D的mat3对象中提取每一行
    row0 = pdmat3.getRow(0)
    row1 = pdmat3.getRow(1)
    row2 = pdmat3.getRow(2)
    # 返回一个NumPy数组,结构为3x3矩阵
    return np.array([[row0[0], row1[0], row2[0]], [row0[1], row1[1], row2[1]], [row0[2], row1[2], row2[2]]])


def npv3mat3_to_pdmat4(npvec3=np.array([0, 0, 0]), npmat3=np.eye(3)):
    """
    将一个3x3的NumPy矩阵和一个1x3的向量转换为Panda3D中的LMatrix4对象

    其中第一个参数是旋转矩阵,第二个是平移向量,如果不传入平移向量,将使用默认值[0, 0, 0]

    :param npmat3: 3x3的NumPy矩阵,表示旋转
    :param npvec3: 1x3的NumPy数组,表示平移
    :return: 转换后的LMatrix4对象,包含旋转和平移信息
    作者: weiwei
    日期: 20170322
    """
    return Mat4(npmat3[0, 0], npmat3[1, 0], npmat3[2, 0], 0, \
                npmat3[0, 1], npmat3[1, 1], npmat3[2, 1], 0, \
                npmat3[0, 2], npmat3[1, 2], npmat3[2, 2], 0, \
                npvec3[0], npvec3[1], npvec3[2], 1)


def npmat4_to_pdmat4(npmat4):
    """
    将一个4x4的NumPy矩阵转换为Panda3D中的LMatrix4对象

    :param npmat4: 4x4的NumPy矩阵
    :return: 转换后的LMatrix4对象

    作者: weiwei
    日期: 20170322
    """
    return Mat4(npmat4[0, 0], npmat4[1, 0], npmat4[2, 0], 0, \
                npmat4[0, 1], npmat4[1, 1], npmat4[2, 1], 0, \
                npmat4[0, 2], npmat4[1, 2], npmat4[2, 2], 0, \
                npmat4[0, 3], npmat4[1, 3], npmat4[2, 3], 1)


def pdmat4_to_npmat4(pdmat4):
    """
    将Panda3D的mat4矩阵转换为NumPy的二维数组

    :param pdmat4: Panda3D中的mat4矩阵
    :return: 对应的NumPy二维数组

    作者: weiwei
    日期: 20161216札幌
    """
    # 获取Panda3D的mat4矩阵的每一行,并转置为NumPy数组
    return np.array(pdmat4.getRows()).T


def pdmat4_to_npv3mat3(pdmat4):
    """
    将Panda3D的mat4矩阵提取出位置和平移矩阵

    :param pdmat4: Panda3D的mat4矩阵
    :return: 返回位置向量和旋转矩阵,分别是1x3和3x3的NumPy数组

    作者: weiwei
    日期: 20200206
    """
    # 获取Panda3D的mat4矩阵的每一行,并转置为NumPy数组
    homomat = np.array(pdmat4.getRows()).T
    # 返回平移部分(前3行的最后一列)和旋转部分(前3行的前3列)
    return [homomat[:3, 3], homomat[:3, :3]]


def npmat3_to_pdquat(npmat3):
    """
    将一个3x3的NumPy矩阵转换为Panda3D中的四元数

    :param npmat3: 3x3的NumPy矩阵,表示旋转
    :return: 转换后的LQuaternion四元数对象

    作者: weiwei
    日期: 20210109
    """
    # 创建一个LQuaternion对象
    quat = LQuaternion()
    # 将旋转矩阵设置为四元数
    quat.setFromMatrix(npmat3_to_pdmat3(npmat3))
    return quat


def pdquat_to_npmat3(pdquat):
    """
    将Panda3D的LQuaternion(四元数)转换为NumPy的3x3矩阵

    :param pdquat: Panda3D的LQuaternion对象
    :return: 转换后的3x3的NumPy矩阵

    作者: weiwei
    日期: 20210109
    """
    # 创建一个空的Mat3对象
    tmp_pdmat3 = Mat3()
    # 将四元数转换为旋转矩阵
    pdquat.extractToMatrix(tmp_pdmat3)
    # 将Mat3对象转换为NumPy数组并返回
    return pdmat3_to_npmat3(tmp_pdmat3)


def npv3_to_pdv3(npv3):
    """
    将NumPy数组转换为Panda3D的Vec3对象

    :param npv3: 一个包含3个元素的NumPy数组
    :return: 对应的Panda3D Vec3对象

    作者: weiwei
    日期: 20170322
    """
    # 将NumPy数组转换为Vec3对象
    return Vec3(npv3[0], npv3[1], npv3[2])


def pdv3_to_npv3(pdv3):
    """
    将Panda3D的Vec3对象转换为NumPy数组

    :param pdv3: Panda3D的Vec3对象
    :return: 对应的NumPy 1D数组

    作者: weiwei
    日期: 20161216札幌
    """
    # 从Panda3D的Vec3对象中提取值并转换为NumPy数组
    return np.array([pdv3[0], pdv3[1], pdv3[2]])


def npv4_to_pdv4(npv4):
    """
    将NumPy数组转换为Panda3D的Vec4对象

    :param npv4: 一个包含4个元素的NumPy数组
    :return: 对应的Panda3D Vec4对象

    作者: weiwei
    日期: 20170322
    """
    # 将NumPy数组转换为Vec4对象
    return Vec4(npv4[0], npv4[1], npv4[2], npv4[3])


def pdv4_to_npv4(pdv4):
    """
    将Panda3D的Vec4对象转换为NumPy数组

    :param pdv4: Panda3D的Vec4对象
    :return: 对应的NumPy 1D数组

    作者: weiwei
    日期: 20161216札幌
    """
    # 从Panda3D的Vec4对象提取每个分量并返回为NumPy数组
    return np.array([pdv4[0], pdv4[1], pdv4[2], pdv4[3]])


def trimesh_to_nodepath(trimesh, name="auto"):
    """
    将三角形网格模型转换为Panda3D的NodePath模型

    :param trimesh: 输入的三角形网格模型
    :param name: 可选的模型名称,默认使用"auto"
    :return: 转换后的NodePath对象

    作者: weiwei
    日期: 20180606
    """
    # 调用nodepath_from_vfnf函数将三角形网格转换为Panda3D的NodePath模型
    return nodepath_from_vfnf(trimesh.vertices, trimesh.face_normals, trimesh.faces, name=name)


def o3dmesh_to_nodepath(o3dmesh, name="auto"):
    """
    将Open3D的网格模型转换为Panda3D的NodePath模型

    :param o3dmesh: 输入的Open3D网格模型
    :param name: 可选的模型名称,默认使用"auto"
    :return: 转换后的NodePath对象

    作者: weiwei
    日期: 20191210
    """
    # 调用nodepath_from_vfnf函数将Open3D网格模型转换为Panda3D的NodePath模型
    return nodepath_from_vfnf(o3dmesh.vertices, o3dmesh.triangle_normals, o3dmesh.triangles, name=name)


def pandageom_from_vfnf(vertices, face_normals, triangles, name='auto'):
    """
    从顶点、面法线和三角形数据创建一个Panda3D的几何模型

    :param vertices: nx3的NumPy数组,每行表示一个顶点
    :param face_normals: nx3的NumPy数组,每行表示一个面的法线
    :param triangles: nx3的NumPy数组,每行表示三个顶点的索引
    :param name: 几何模型的名称,默认为"auto"
    :return: 创建的Panda3D几何模型(Geom对象)

    作者: weiwei
    日期: 20160613, 20210109
    """
    # 扩展顶点数据,使每个三角形都引用不同的顶点和法线
    # 顶点和法线格式
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    # 将三角形的顶点索引展平并生成顶点数据
    vertids = triangles.flatten()
    multiplied_verticies = np.empty((len(vertids), 3), dtype=np.float32)
    multiplied_verticies[:] = vertices[vertids]
    # 为每个三角形的顶点重复对应的法线
    vertex_normals = np.repeat(face_normals.astype(np.float32), repeats=3, axis=0)
    # 将顶点和法线合并为一维字节数组
    npstr = np.hstack((multiplied_verticies, vertex_normals)).tobytes()
    vertexdata.modifyArrayHandle(0).setData(npstr)
    # 三角形数据
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    multiplied_triangles = np.arange(len(vertids), dtype=np.uint32).reshape(-1, 3)
    primitive.modifyVertices(-1).modifyHandle().setData(multiplied_triangles.tobytes())
    # 创建几何体并添加三角形数据
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vfnf(vertices, face_normals, triangles, name=''):
    """
    将给定的顶点和三角形数据打包成 Panda3D 的几何体(geom)并返回NodePath对象

    :param vertices: nx3的NumPy数组,每行表示一个顶点
    :param face_normals: nx3的NumPy数组,每行表示一个面的法线
    :param triangles: nx3的NumPy数组,每行表示三角形的顶点索引
    :param name: 可选的名称,用于命名生成的对象
    :return: 返回一个包含几何体的Panda3D NodePath对象

    作者: weiwei
    日期: 20170221, 20210109
    """
    # 从顶点、法线和三角形数据创建几何体
    objgeom = pandageom_from_vfnf(vertices, face_normals, triangles, name + 'geom')
    # 创建GeomNode对象
    geomnodeobj = GeomNode(name + 'geomnode')
    # 将几何体添加到GeomNode
    geomnodeobj.addGeom(objgeom)
    # 创建NodePath对象并将GeomNode附加到NodePath
    pandanp = NodePath(name)
    pandanp.attachNewNode(geomnodeobj)
    return pandanp


def pandageom_from_vvnf(vertices, vertex_normals, triangles, name=''):
    """
    使用环境中的 collisionmodel 代替,将顶点、顶点法线和三角形数据打包成Panda3D的几何体(geom)

    :param vertices: nx3的NumPy数组,每行表示一个顶点
    :param vertex_normals: nx3的NumPy数组,每行表示一个顶点的法线
    :param triangles: nx3的NumPy数组,每行表示三角形的顶点索引
    :param name: 几何体的名称
    :return: 创建的Panda3D几何体(Geom对象)

    作者: weiwei
    日期: 20171219, 20210901
    """
    # 创建顶点和法线数据格式
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertexdata.modifyArrayHandle(0).setData(np.hstack((vertices, vertex_normals)).astype(np.float32).tobytes())
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().setData(triangles.astype(np.uint32).tobytes())
    # make geom
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vvnf(vertices, vertnormals, triangles, name=''):
    """
    使用环境中的 collisionmodel 代替,将顶点、顶点法线和三角形数据打包成Panda3D的NodePath对象

    :param vertices: nx3的NumPy数组,每行表示一个顶点
    :param vertnormals: nx3的NumPy数组,每行表示一个顶点的法线
    :param triangles: nx3的NumPy数组,每行表示三角形的顶点索引
    :param name: 可选的名称,用于命名生成的对象
    :return: 返回一个包含几何体的Panda3D NodePath对象

    作者: weiwei
    日期: 20170221, 20210109
    """
    # 从顶点、法线和三角形数据创建几何体
    objgeom = pandageom_from_vvnf(vertices, vertnormals, triangles, name + 'geom')
    geomnodeobj = GeomNode('GeomNode')
    geomnodeobj.addGeom(objgeom)
    pandanp = NodePath(name + 'nodepath')
    pandanp.attachNewNode(geomnodeobj)
    return pandanp


# def pandageom_from_points(vertices, rgba=None, name=''):
#     """
#     将顶点数据打包成Panda3D的点云几何体(geom)
#
#     :param vertices: 顶点数据,应该是一个 nx3 的 NumPy 数组
#     :param rgba: 每个顶点的颜色信息,可以是单一颜色或每个顶点不同的颜色
#     :param name: 几何体的名称
#     :return: Panda3D的点云几何体
#     """
#     # 创建顶点数据结构
#     vertex_format = GeomVertexFormat.get_v3c4()
#     vertex_data = GeomVertexData(name, vertex_format, Geom.UH_static)
#     vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
#     color_writer = GeomVertexWriter(vertex_data, 'color')
#
#     # 处理颜色信息
#     if rgba is None:
#         # 默认颜色为黑色,透明度为1
#         vertex_rgbas = np.asarray([[0, 0, 0, 1]] * len(vertices))
#     else:
#         if isinstance(rgba, list) or isinstance(rgba, tuple):
#             rgba = np.array(rgba)
#         if not isinstance(rgba, np.ndarray):
#             raise ValueError('rgba must be a list or an nparray!')
#
#         if rgba.ndim == 1:
#             # 单一颜色,扩展到与顶点数量相同
#             vertex_rgbas = np.tile((rgba * 255).astype(np.uint8), (len(vertices), 1))
#         elif rgba.ndim == 2 and rgba.shape[0] == len(vertices):
#             # 多种颜色,每个顶点一个颜色
#             vertex_rgbas = (rgba * 255).astype(np.uint8)
#         else:
#             raise ValueError('rgba must be a single color or match the number of vertices!')
#
#     # 写入顶点和颜色数据
#     for i, vertex in enumerate(vertices):
#         vertex_writer.addData3f(*vertex)
#         color_writer.addData4f(*vertex_rgbas[i])
#
#     # 创建点云几何体
#     primitive = GeomPoints(Geom.UH_static)
#     primitive.add_next_vertices(len(vertices))
#     geom = Geom(vertex_data)
#     geom.addPrimitive(primitive)
#
#     return geom

def pandageom_from_points(vertices, rgba=None, name=''):
    """
    将顶点数据打包成Panda3D的点云几何体(geom)

    :param vertices: 顶点数据,应该是一个 nx3 的 NumPy 数组
    :param rgba: 每个顶点的RGBA颜色,可以是列表或NumPy数组(RGB颜色也可以)
    :param name: 几何体的名称
    :return: 返回一个Panda3D的几何体(Geom对象)

    作者: weiwei
    日期: 20170328, 20210116, 20220721
    """
    # if rgba is None:
    #     vertex_rgbas = np.asarray([[0, 0, 0, 1]] * len(vertices))
    #     # vertex_rgbas = np.asarray([[0, 0, 0, 255]] * len(vertices), dtype=np.uint8)
    # if isinstance(rgba, list) or isinstance(rgba, tuple):
    #     rgba = np.array(rgba)
    # if not isinstance(rgba, np.ndarray):
    #     raise ValueError('rgba must be a list or an nparray!')
    # if len(rgba) == 1:
    #     vertex_rgbas = np.tile((rgba * 255).astype(np.uint8), (len(vertices), 1))
    # elif len(rgba) == len(vertices):
    #     vertex_rgbas = (rgba * 255).astype(np.uint8)
    # n_color_bit = rgba.shape[1]

    num_vertices = len(vertices)

    # 处理颜色数据
    if rgba is None:
        # 如果没有提供颜色,使用默认黑色
        vertex_rgbas = np.asarray([[0, 0, 0, 1]] * num_vertices)
    else:
        if isinstance(rgba, (list, tuple)):
            rgba = np.array(rgba)
        if not isinstance(rgba, np.ndarray):
            raise ValueError('rgba must be a list or an nparray!')
        if len(rgba) == 1:
            # 单一颜色,复制到所有顶点
            vertex_rgbas = np.tile(rgba, (num_vertices, 1))
        elif len(rgba) == num_vertices:
            vertex_rgbas = rgba
        else:
            raise ValueError('rgba length must be 1 or equal to the number of vertices')

    # 确保颜色值在 0 到 1 的范围内,并转换为 0 到 255 的整数
    vertex_rgbas = (vertex_rgbas * 255).astype(np.uint8)
    n_color_bit = vertex_rgbas.shape[1]

    # 创建顶点数据格式
    vertformat = GeomVertexFormat()
    arrayformat = GeomVertexArrayFormat()
    arrayformat.addColumn(InternalName.getVertex(), 3, GeomEnums.NTFloat32, GeomEnums.CPoint)
    vertformat.addArray(arrayformat)

    # 添加颜色通道
    arrayformat = GeomVertexArrayFormat()
    arrayformat.addColumn(InternalName.getColor(), n_color_bit, GeomEnums.NTUint8, GeomEnums.CColor)
    vertformat.addArray(arrayformat)

    # 注册格式
    vertformat = GeomVertexFormat.registerFormat(vertformat)

    # 创建顶点数据并填充顶点和颜色数据
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertexdata.modifyArrayHandle(0).copyDataFrom(np.ascontiguousarray(vertices, dtype=np.float32))
    vertexdata.modifyArrayHandle(1).copyDataFrom(vertex_rgbas)

    # 创建几何体(点云)
    primitive = GeomPoints(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().copyDataFrom(np.arange(len(vertices), dtype=np.uint32))

    # 创建几何体对象并添加到几何体中
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_points(vertices, rgba_list=None, name=''):
    """
    将顶点数据打包成Panda3D的点云节点路径(NodePath)

    :param vertices: 顶点数据,应该是一个 nx3 的 NumPy 数组
    :param rgba_list: 一个包含单一1x4 NumPy数组的列表,或者长度与顶点数相同的1x4 NumPy数组列表
    :param name: 节点路径的名称
    :return: 返回一个Panda3D的NodePath对象

    作者: weiwei
    日期: 20170328
    """
    if rgba_list is None:
        rgba_list = np.array([[1, 0, 0, 1]] * len(vertices))  # 默认颜色
    elif isinstance(rgba_list, list) or isinstance(rgba_list, tuple):
        rgba_list = np.array(rgba_list)

    if rgba_list.shape[0] != 1 and rgba_list.shape[0] != len(vertices):
        raise ValueError('rgba length must be 1 or equal to the number of vertices')

    # 从顶点数据创建几何体
    objgeom = pandageom_from_points(vertices, rgba_list, name + 'geom')

    # 创建GeomNode对象并添加几何体
    geomnodeobj = GeomNode('GeomNode')
    geomnodeobj.addGeom(objgeom)

    # 创建NodePath对象并关闭光照
    pointcloud_nodepath = NodePath(name)
    pointcloud_nodepath.setLightOff()
    pointcloud_nodepath.attachNewNode(geomnodeobj)
    return pointcloud_nodepath


def loadfile_vf(objpath):
    """
    加载网格对象到 Panda3D 的 NodePath 中使用面法线来打包数据

    :param objpath: 网格文件路径
    :return: 返回Panda3D的NodePath对象

    作者: weiwei
    日期: 20170221
    """
    objtrm = trm.load_mesh(objpath)
    pdnp = nodepath_from_vfnf(objtrm.vertices, objtrm.face_normals, objtrm.faces)
    return pdnp


def loadfile_vvnf(objpath):
    """
    加载网格对象到Panda3D的NodePath中,使用顶点法线来打包数据

    :param objpath: 网格文件路径
    :return: 返回Panda3D的NodePath对象

    作者: weiwei
    日期: 20170221
    """
    objtrm = trm.load_mesh(objpath)
    pdnp = nodepath_from_vvnf(objtrm.vertices, objtrm.vertex_normals, objtrm.faces)
    return pdnp


if __name__ == '__main__':
    import os, math, basis
    import basis.trimesh as trimesh
    import visualization.panda.world as wd
    from panda3d.core import TransparencyAttrib

    base = wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bt = trimesh.load(objpath)
    btch = bt.convex_hull
    pdnp = nodepath_from_vfnf(bt.vertices, bt.face_normals, bt.faces)
    pdnp.reparentTo(base.render)
    pdnp_cvxh = nodepath_from_vfnf(btch.vertices, btch.face_normals, btch.faces)
    pdnp_cvxh.setTransparency(TransparencyAttrib.MDual)
    pdnp_cvxh.setColor(0, 1, 0, .3)
    pdnp_cvxh.reparentTo(base.render)
    pdnp2 = nodepath_from_vvnf(bt.vertices, bt.vertex_normals, bt.faces)
    pdnp2.setPos(0, 0, .1)
    pdnp2.reparentTo(base.render)
    base.run()
