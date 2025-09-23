import numpy as np
from ..base import Trimesh
from ..constants import _log_time, log
from ..util import is_file, is_string, make_sequence, is_instance_named, concatenate
from .assimp import _assimp_loaders
from .stl import _stl_loaders
from .misc import _misc_loaders
from .step import _step_loaders
from .ply import _ply_loaders
from .dae import _collada_loaders

try:
    from ..path.io.load import load_path, path_formats
except:
    log.warning('没有可用的路径功能!', exc_info=True)


def load_path(*args, **kwargs):
    # 抛出异常,表示没有可用的路径功能
    raise ImportError('没有可用的路径功能!')


def path_formats():
    # 返回一个空列表,表示没有可用的路径格式
    return []


def mesh_formats():
    # 返回支持的网格格式列表
    return list(_mesh_loaders.keys())


def available_formats():
    # 返回可用的格式,包括网格和路径格式
    return np.append(mesh_formats(), path_formats())


def load(obj, file_type=None, **kwargs):
    """
    加载网格或矢量路径到 Trimesh、Path2D 或 Path3D 对象中

    :param obj: 文件名字符串或文件对象
    :param file_type: 字符串,表示文件类型(例如:'meshes'),如果为 None 则从 obj 中解析
    :param kwargs: 其他参数
    :return: Trimesh、Path2D、Path3D 或其列表

    author: revised by weiwei
    date: 20201206
    """
    if isinstance(obj, Trimesh):
        return obj
    if is_instance_named(obj, 'Path'):
        return obj
    if is_string(obj):
        file_type = (str(obj).split('.')[-1]).lower()
        obj = open(obj, 'rb')
    if file_type is None:
        file_type = obj.__class__.__name__
    if file_type in path_formats():
        return load_path(obj, file_type, **kwargs)
    elif file_type in mesh_formats():
        return load_mesh(obj, file_type, **kwargs)
    raise ValueError('File type: %s not supported', str(file_type))


@_log_time
def load_mesh(obj, file_type=None, process=True):
    '''
    加载网格文件到 Trimesh 对象中

    :param obj: str 或 文件对象,包含网格数据的文件名字符串或文件对象
    :param file_type: str, 可选,表示文件类型的字符串(例如:'meshes')如果为 None,则从文件扩展名或对象类型推断文件类型
    :param process: bool, 可选,布尔值,指示是否在加载时处理网格.默认值为 True
    :returns mesh:单个 Trimesh 对象或 Trimesh 对象列表,具体取决于文件格式

    author: revised by weiwei
    date: 20220212
    '''
    if is_string(obj):
        file_type = (str(obj).split('.')[-1]).lower()
        obj = open(obj, 'rb')
    if file_type is None:
        file_type = obj.__class__.__name__
    file_type = str(file_type).lower()
    loaded = _mesh_loaders[file_type](obj, file_type)
    if is_file(obj):
        obj.close()
    log.debug('loaded mesh using %s', _mesh_loaders[file_type].__name__)
    meshes = [Trimesh(process=process, **i) for i in make_sequence(loaded)]
    # if len(meshes) == 1:
    #     return meshes[0]
    meshes = concatenate(meshes)
    return meshes


_mesh_loaders = {}
# assimp 有很多加载器,但它们都很慢
# 所以我们先加载它们,然后如果可能的话用本地加载器替换
_mesh_loaders.update(_assimp_loaders)
_mesh_loaders.update(_stl_loaders)
_mesh_loaders.update(_misc_loaders)
_mesh_loaders.update(_step_loaders)
_mesh_loaders.update(_ply_loaders)
_mesh_loaders.update(_collada_loaders)
