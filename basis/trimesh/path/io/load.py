import numpy as np
import os
from .dxf_load import load_dxf
from .svg_load import svg_to_path
from .misc import lines_to_path, polygon_to_path, dict_to_path
from ..path import Path, Path2D, Path3D
from ...util import is_sequence, is_file, is_string, is_instance_named


def load_path(obj, file_type=None):
    '''
    工具函数,可以传入文件名、文件对象或行列表,并返回 Path 对象.

    :param obj: 可以是文件名(字符串)、文件对象、Path 对象、Polygon 对象、字典或行列表
    :param file_type: 文件类型字符串(例如 'dxf', 'svg'),如果 obj 是文件名或文件对象时使用
    :return: Path 对象
    '''
    if isinstance(obj, Path):
        return obj
    elif is_file(obj):
        # 如果是文件对象,使用指定的加载器加载
        loaded = _LOADERS[file_type](obj)
        obj.close()
    elif is_string(obj):
        # 如果是文件名字符串,打开文件并确定文件类型
        file_obj = open(obj, 'rb')
        file_type = os.path.splitext(obj)[-1][1:].lower()
        loaded = _LOADERS[file_type](file_obj)
        file_obj.close()
    elif is_instance_named(obj, 'Polygon'):
        # 如果是 Polygon 对象,转换为 Path
        loaded = polygon_to_path(obj)
    elif is_instance_named(obj, 'dict'):
        # 如果是字典,转换为 Path
        loaded = dict_to_path(obj)
    elif is_sequence(obj):
        # 如果是行列表,转换为 Path
        loaded = lines_to_path(obj)
    else:
        raise ValueError('不支持的对象类型！')
    path = _create_path(**loaded)
    return path


def _create_path(entities, vertices, metadata=None):
    '''
    创建 Path2D 或 Path3D 对象

    :param entities: 实体列表
    :param vertices: 顶点数组,必须是 2D 或 3D
    :param metadata: 可选的元数据字典
    :return: Path2D 或 Path3D 对象
    '''
    shape = np.shape(vertices)
    if ((len(shape) != 2) or
            (not shape[1] in [2, 3])):
        raise ValueError('顶点必须是 2D 或 3D！')
    path = [Path2D, Path3D][shape[1] == 3](entities=entities,
                                           vertices=vertices,
                                           metadata=metadata)
    return path


def path_formats():
    '''
    返回支持的路径格式列表

    :return: 支持的路径格式列表
    '''
    return list(_LOADERS.keys())


_LOADERS = {'dxf': load_dxf,
            'svg': svg_to_path}
