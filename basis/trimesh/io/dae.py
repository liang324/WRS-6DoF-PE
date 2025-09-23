# 注意:此加载器假定在dae文件20201207中没有转换

import io
import uuid
import numpy as np
from .. import transform_points

try:
    # pip install pycollada
    import collada
except BaseException:
    collada = None
try:
    import PIL.Image
except ImportError:
    pass
from .. import util
from .. import visual
from ..constants import log


def load_collada(file_obj, resolver=None, **kwargs):
    """
    加载 COLLADA (.dae) 文件并转换为 trimesh 的参数列表

    :param file_obj: 文件对象,包含COLLADA文件
    :param resolver: trimesh.visual.Resolver 或 None,用于加载引用文件,例如纹理图像
    :param kwargs: 其他参数,传递给trimesh.Trimesh.__init__
    :return: 用于Trimesh构造函数的参数字典列表
    """
    # 使用pycollada加载场景
    c = collada.Collada(file_obj)
    # 创建从 Material ID 到 trimesh 材质的映射
    material_map = {}
    # for m in c.materials:
    #     effect = m.effect
    #     material_map[m.id] = _parse_material(effect, resolver)
    # name : kwargs

    # 遍历场景节点
    meshes = {}
    graph = []
    for node in c.scene.nodes:
        _parse_node(node=node,
                    parent_matrix=np.eye(4),
                    material_map=material_map,
                    meshes=meshes,
                    graph=graph,
                    resolver=resolver)
    # create kwargs for load_kwargs
    # result = {'class': 'Scene',
    #           'graph': graph,
    #           'geometry': meshes}
    # 返回网格参数列表
    return list(meshes.values())


def export_collada(mesh, **kwargs):
    """
    将一个或多个网格导出为COLLADA (.dae) 文件

    :param mesh: Trimesh对象或Trimesh对象的列表 要导出的网格
    :return: str,COLLADA格式的输出字符串
    """
    meshes = mesh
    if not isinstance(mesh, (list, tuple, set, np.ndarray)):
        meshes = [mesh]
    c = collada.Collada()
    nodes = []
    for i, m in enumerate(meshes):
        # 加载 UV 坐标、颜色和材质
        uv = None
        colors = None
        mat = _unparse_material(None)
        if m.visual.defined:
            if m.visual.kind == 'texture':
                mat = _unparse_material(m.visual.material)
                uv = m.visual.uv
            elif m.visual.kind == 'vertex':
                colors = (m.visual.vertex_colors / 255.0)[:, :3]
        c.effects.append(mat.effect)
        c.materials.append(mat)

        # 创建几何对象
        vertices = collada.source.FloatSource('verts-array', m.vertices.flatten(), ('X', 'Y', 'Z'))
        normals = collada.source.FloatSource('normals-array', m.vertex_normals.flatten(), ('X', 'Y', 'Z'))
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', '#verts-array')
        input_list.addInput(1, 'NORMAL', '#normals-array')
        arrays = [vertices, normals]
        if uv is not None:
            texcoords = collada.source.FloatSource('texcoords-array', uv.flatten(), ('U', 'V'))
            input_list.addInput(2, 'TEXCOORD', '#texcoords-array')
            arrays.append(texcoords)
        if colors is not None:
            idx = 2
            if uv:
                idx = 3
            colors = collada.source.FloatSource('colors-array', colors.flatten(), ('R', 'G', 'B'))
            input_list.addInput(idx, 'COLOR', '#colors-array')
            arrays.append(colors)

        geom = collada.geometry.Geometry(c, uuid.uuid4().hex, uuid.uuid4().hex, arrays)
        indices = np.repeat(m.faces.flatten(), len(arrays))
        matref = 'material{}'.format(i)
        triset = geom.createTriangleSet(indices, input_list, matref)
        geom.primitives.append(triset)
        c.geometries.append(geom)
        matnode = collada.scene.MaterialNode(matref, mat, inputs=[])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node = collada.scene.Node('node{}'.format(i), children=[geomnode])
        nodes.append(node)

    scene = collada.scene.Scene('scene', nodes)
    c.scenes.append(scene)
    c.scene = scene
    b = io.BytesIO()
    c.write(b)
    b.seek(0)
    return b.read()


def _parse_node(node,
                parent_matrix,
                material_map,
                meshes,
                graph,
                resolver=None):
    """
    递归解析 COLLADA 场景节点

    :param node: 当前解析的节点
    :param parent_matrix: 父节点的变换矩阵
    :param material_map: 材质映射字典
    :param meshes: 存储解析结果的网格字典
    :param graph: 场景图信息
    :param resolver: 用于解析外部资源的解析器
    """
    # 解析几何节点
    if isinstance(node, collada.scene.GeometryNode):
        geometry = node.geometry
        # 遍历几何体的原语
        for i, primitive in enumerate(geometry.primitives):
            if isinstance(primitive, collada.polylist.Polylist):
                primitive = primitive.triangleset()
            if isinstance(primitive, collada.triangleset.TriangleSet):
                vertices = primitive.vertex
                faces = primitive.vertex_index
                normal = primitive.normal
                vertex_normals = normal[primitive.normal_index]
                face_normals = (vertex_normals[:, 0, :] + vertex_normals[:, 1, :] + vertex_normals[:, 2, :]) / 3

                if not np.allclose(parent_matrix, np.eye(4), 1e-8):
                    vertices = transform_points(vertices, parent_matrix)
                    normalized_matrix = parent_matrix / np.linalg.norm(parent_matrix[:, 0])
                    face_normals = transform_points(face_normals, normalized_matrix, translate=False)
                primid = '{}.{}'.format(geometry.id, i)
                meshes[primid] = {
                    'vertices': vertices,
                    'faces': faces,
                    'face_normals': face_normals}
                graph.append({'frame_to': primid,
                              'matrix': parent_matrix,
                              'geometry': primid})

    # 递归解析子节点
    elif isinstance(node, collada.scene.Node):
        if node.children is not None:
            for child in node.children:
                # 创建新的变换矩阵
                matrix = np.dot(parent_matrix, node.matrix)
                # 解析子节点
                _parse_node(
                    node=child,
                    parent_matrix=matrix,
                    material_map=material_map,
                    meshes=meshes,
                    graph=graph,
                    resolver=resolver)
    elif isinstance(node, collada.scene.CameraNode):
        # TODO: 将 COLLADA相机转换为 trimesh相机
        pass
    elif isinstance(node, collada.scene.LightNode):
        # TODO: 将 COLLADA灯光转换为trimesh灯光
        pass


def _load_texture(file_name, resolver):
    """
    从文件中加载纹理并转换为 PIL 图像

    :param file_name: str,纹理文件的名称
    :param resolver: trimesh.visual.Resolver,用于解析文件路径
    :return: PIL.Image,加载的图像
    """
    file_data = resolver.get(file_name)  # 使用解析器获取文件数据
    image = PIL.Image.open(util.wrap_as_stream(file_data))  # 打开图像文件
    return image  # 返回图像


def _parse_material(effect, resolver):
    """
    将 COLLADA效果转换为 trimesh 材质

    :param effect: collada.material.Effect,COLLADA材质效果
    :param resolver: trimesh.visual.Resolver,用于解析纹理文件
    :return: visual.material.PBRMaterial,trimesh材质
    """
    # 计算基础颜色
    baseColorFactor = np.ones(4)
    baseColorTexture = None
    if isinstance(effect.diffuse, collada.material.Map):
        try:
            baseColorTexture = _load_texture(effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('无法加载基础纹理', exc_info=True)
    elif effect.diffuse is not None:
        baseColorFactor = effect.diffuse

    # 计算发射颜色
    emissiveFactor = np.zeros(3)
    emissiveTexture = None
    if isinstance(effect.emission, collada.material.Map):
        try:
            emissiveTexture = _load_texture(effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('无法加载 emissive 纹理', exc_info=True)
    elif effect.emission is not None:
        emissiveFactor = effect.emission[:3]

    # 计算粗糙度
    roughnessFactor = 1.0
    if (not isinstance(effect.shininess, collada.material.Map) and effect.shininess is not None):
        roughnessFactor = np.sqrt(2.0 / (2.0 + effect.shininess))

    # 计算金属度
    metallicFactor = 0.0

    # 计算法线纹理
    normalTexture = None
    if effect.bumpmap is not None:
        try:
            normalTexture = _load_texture(effect.bumpmap.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('unable to load bumpmap', exc_info=True)
    return visual.material.PBRMaterial(
        emissiveFactor=emissiveFactor,
        emissiveTexture=emissiveTexture,
        normalTexture=normalTexture,
        baseColorTexture=baseColorTexture,
        baseColorFactor=baseColorFactor,
        metallicFactor=metallicFactor,
        roughnessFactor=roughnessFactor)


def _unparse_material(material):
    """
    将 trimesh材质转换为 COLLADA材质

    :param material: visual.material.PBRMaterial,trimesh材质
    :return: collada.material.Material,COLLADA材质
    """
    # TODO: 导出纹理
    if isinstance(material, visual.material.PBRMaterial):
        diffuse = material.baseColorFactor
        if diffuse is not None:
            diffuse = list(diffuse)
        emission = material.emissiveFactor
        if emission is not None:
            emission = [float(emission[0]), float(emission[1]), float(emission[2]), 1.0]
        shininess = material.roughnessFactor
        if shininess is not None:
            shininess = 2.0 / shininess ** 2 - 2.0
        effect = collada.material.Effect(
            uuid.uuid4().hex, params=[], shadingtype='phong',
            diffuse=diffuse, emission=emission,
            specular=[1.0, 1.0, 1.0, 1.0], shininess=float(shininess)
        )
        material = collada.material.Material(uuid.uuid4().hex, 'pbrmaterial', effect)
    else:
        effect = collada.material.Effect(uuid.uuid4().hex, params=[], shadingtype='phong')
        material = collada.material.Material(uuid.uuid4().hex, 'defaultmaterial', effect)
    return material


def load_zae(file_obj, resolver=None, **kwargs):
    """
    加载 ZAE文件,这实际上是一个压缩的DAE文件

    :param file_obj: 文件对象,包含ZAE数据
    :param resolver: trimesh.visual.Resolver,用于加载附加资源
    :param kwargs: dict,传递给load_collada的其他参数
    :return: dict,加载结果
    """
    # 解压缩文件,得到一个字典 {文件名: 文件对象}
    archive = util.decompress(file_obj, file_type='zip')
    # 加载第一个具有.dae扩展名的文件
    file_name = next(i for i in archive.keys() if i.lower().endswith('.dae'))
    # 创建解析器以加载纹理等资源
    resolver = visual.resolvers.ZipResolver(archive)
    # 使用常规的collada加载器
    loaded = load_collada(archive[file_name], resolver=resolver, **kwargs)
    return loaded


# 只有在安装了`pycollada`后才提供loader
_collada_loaders = {}
_collada_exporters = {}
if collada is not None:
    _collada_loaders['dae'] = load_collada
    _collada_loaders['zae'] = load_zae
    _collada_exporters['dae'] = export_collada
