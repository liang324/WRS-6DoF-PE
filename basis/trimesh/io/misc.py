import numpy as np
from ..util import encoded_to_array


def load_off(file_obj, file_type=None):
    """
    加载 OFF 文件并返回包含顶点和面信息的字典

    :param file_obj: 文件对象,包含 OFF 数据
    :param file_type: str, optional,文件类型(默认为 None)
    :return: dict,包含顶点和面信息的字典
    """
    header_string = file_obj.readline().decode().strip()
    if not header_string == 'OFF':
        raise NameError('不是一个 OFF 文件！头部信息是' + header_string)

    header = np.array(file_obj.readline().split()).astype(int)
    blob = np.array(file_obj.read().split())
    data_ok = np.sum(header * [3, 4, 0]) == len(blob)
    if not data_ok: raise NameError('顶点或面的数量不正确！')

    vertices = blob[0:(header[0] * 3)].astype(float).reshape((-1, 3))
    faces = blob[(header[0] * 3):].astype(int).reshape((-1, 4))[:, 1:]

    return {'vertices': vertices, 'faces': faces}


def load_wavefront(file_obj, file_type=None):
    '''
    加载 Wavefront .obj 文件到 Trimesh 对象中

    丢弃纹理法线和顶点颜色信息
    参考:https://en.wikipedia.org/wiki/Wavefront_.obj_file
    '''
    data = np.array(file_obj.read().split())
    data_str = data.astype(str)

    # 找到关键字的位置,然后找到后续的值
    vid = np.nonzero(data_str == 'v')[0].reshape((-1, 1)) + np.arange(3) + 1
    nid = np.nonzero(data_str == 'vn')[0].reshape((-1, 1)) + np.arange(3) + 1
    fid = np.nonzero(data_str == 'f')[0].reshape((-1, 1)) + np.arange(3) + 1

    # 如果我们想使用纹理/顶点法线,可以以不同方式切片
    faces = np.array([i.split(b'/') for i in data[fid].reshape(-1)])[:, 0].reshape((-1, 3))
    # Wavefront 使用 1 索引的面,而不是 0 索引
    faces = faces.astype(int) - 1
    return {'vertices': data[vid].astype(float),
            'vertex_normals': data[nid].astype(float),
            'faces': faces}


def load_dict(data, file_type=None):
    """
    加载字典数据

    :param data: 字典数据
    :param file_type: str, optional,文件类型(默认为 None)
    :return: 原始字典数据
    """
    return data


def load_dict64(data, file_type=None):
    """
    加载 64 位编码的字典数据,并解码为数组

    :param data: 64 位编码的字典数据
    :param file_type: str, optional,文件类型(默认为 None)
    :return: 解码后的字典数据
    """
    for key in ('vertices', 'faces', 'face_normals'):
        data[key] = encoded_to_array(data[key])
    return data


_misc_loaders = {'obj': load_wavefront,  # Wavefront .obj 文件加载器
                 'off': load_off,  # OFF 文件加载器
                 'dict': load_dict,  # 字典数据加载器
                 'dict64': load_dict64}  # 64位编码字典数据加载器
