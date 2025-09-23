import numpy as np
from ..util import is_binary_file

_stl_dtype = np.dtype([('normals', np.float32, (3)), ('vertices', np.float32, (3, 3)), ('attributes', np.uint16)])
_stl_dtype_header = np.dtype([('header', np.void, 80), ('face_count', np.int32)])


def load_stl(file_obj, file_type=None):
    '''
    加载 STL 文件并返回包含网格信息的字典

    :param file_obj: 文件对象,包含 STL 数据
    :param file_type: 未使用的参数

    :return: 包含顶点、面和面法线信息的字典
    '''
    if is_binary_file(file_obj):
        return load_stl_binary(file_obj)
    else:
        return load_stl_ascii(file_obj)


def load_stl_binary(file_obj):
    """
    加载二进制 STL 文件到 Trimesh 对象

    使用单个主 struct.unpack 调用,比循环方法或 ASCII STL 快得多

    :param file_obj: 文件对象,包含二进制 STL 数据
    :return: 包含顶点、面和面法线信息的字典
    """
    header = np.fromstring(file_obj.read(84), dtype=_stl_dtype_header)
    # 现在我们检查header的长度与文件的长度
    # data_start应该始终位于位置84,但硬编码让人感觉很难看
    data_start = file_obj.tell()
    # 搜索到文件末尾(相对于` =2 `文件末尾的位置0)
    file_obj.seek(0, 2)
    # 我们保存文件末尾的位置,并返回开始的位置
    data_end = file_obj.tell()
    file_obj.seek(data_start)
    # 二进制格式有一个严格定义的结构,如果文件的长度与头文件不匹配,加载的版本几乎肯定会是垃圾
    data_ok = (data_end - data_start) == (header['face_count'] * _stl_dtype.itemsize)

    # 这个检查是为了检查这是否真的是一个二进制STL文件
    # 如果我们不这样做,并尝试加载一个结构不正确的文件
    # 结构体.Unpack调用使用100%的内存,直到整个程序崩溃,所以最好在这里抛出异常
    if not data_ok:
        raise ValueError('二进制 STL 文件头部长度不正确！')
    # 由于STL格式,我们所有的顶点都将按顺序加载,因此面只是顺序索引的重塑
    faces = np.arange(header['face_count'] * 3).reshape((-1, 3))
    blob = np.fromstring(file_obj.read(), dtype=_stl_dtype)
    result = {'vertices': blob['vertices'].reshape((-1, 3)),
              'face_normals': blob['normals'].reshape((-1, 3)),
              'faces': faces}
    return result


def load_stl_ascii(file_obj):
    '''
    加载 ASCII STL 文件

    :param file_obj: 文件对象,包含 ASCII STL 数据
    :return: 包含顶点、面和面法线信息的字典
    '''
    header = file_obj.readline()
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.lower().split('endsolid')[0]
    blob = np.array(text.split())

    face_len = 21
    face_count = len(blob) / face_len
    if (len(blob) % face_len) != 0:
        raise ValueError('STL 文件中的值数量不正确！')

    face_count = int(face_count)
    # 这个偏移量将被添加到一组固定的平铺索引中
    offset = face_len * np.arange(face_count).reshape((-1, 1))
    normal_index = np.tile([2, 3, 4], (face_count, 1)) + offset
    vertex_index = np.tile([8, 9, 10, 12, 13, 14, 16, 17, 18], (face_count, 1)) + offset

    # 面是由三个连续的顶点组成的组,因为顶点不是引用
    faces = np.arange(face_count * 3).reshape((-1, 3))
    face_normals = blob[normal_index].astype(float)
    vertices = blob[vertex_index.reshape((-1, 3))].astype(float)

    return {'vertices': vertices,
            'faces': faces,
            'face_normals': face_normals}


def export_stl(mesh):
    '''
    将 Trimesh 对象转换为二进制 STL 文件

    :param mesh: Trimesh 对象
    :return: 字节数据,表示二进制 STL 格式的网格
    '''
    header = np.zeros(1, dtype=_stl_dtype_header)
    header['face_count'] = len(mesh.faces)
    packed = np.zeros(len(mesh.faces), dtype=_stl_dtype)
    packed['normals'] = mesh.face_normals
    packed['vertices'] = mesh.triangles
    export = header.tostring()
    export += packed.tostring()
    return export


_stl_loaders = {'stl': load_stl}
