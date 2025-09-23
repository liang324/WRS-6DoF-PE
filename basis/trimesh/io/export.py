import numpy as np
import json
from ..constants import log
from ..util import tolist_dict, is_string, array_to_encoded
from .stl import export_stl
from .ply import export_ply

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


def export_mesh(mesh, file_obj, file_type=None):
    '''
    导出 Trimesh 对象到文件或文件名

    :param mesh: Trimesh 对象,要导出的网格
    :param file_obj: str 或文件对象,目标文件名或文件对象
    :param file_type: str, optional,文件类型(如 'stl', 'obj'),未提供时从文件名推断
    :return: dict,包含导出网格数据的字典
    '''
    if is_string(file_obj):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'wb')
    file_type = str(file_type).lower()
    log.info('Exporting %d faces as %s', len(mesh.faces), file_type.upper())
    export = _mesh_exporters[file_type](mesh)
    if hasattr(file_obj, 'write'):
        file_obj.write(export)
        file_obj.flush()
        file_obj.close()
    else:
        return export


def export_off(mesh):
    # 将网格导出为 OFF 格式
    faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3, mesh.faces)).astype(np.int64)
    export = 'OFF\n'
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += np.array_str(mesh.vertices, precision=9).replace('[', '').replace(']', '').strip()
    export += np.array_str(faces_stacked).replace('[', '').replace(']', '').strip()
    return export


def export_collada(mesh):
    '''
    将网格导出为 COLLADA 文件
    '''
    from ..templates import get_template
    from string import Template

    # 获取 COLLADA 模板字符串
    template_string = get_template('collada.dae.template')
    template = Template(template_string)

    # 设置打印选项,因为 np.array2string 使用这些选项
    np.set_printoptions(threshold=np.inf, precision=5, linewidth=np.inf)
    replacement = dict()
    # 将顶点、面和法线转换为字符串
    replacement['VERTEX'] = np.array2string(mesh.vertices.reshape(-1))[1:-1]
    replacement['FACES'] = np.array2string(mesh.faces.reshape(-1))[1:-1]
    replacement['NORMALS'] = np.array2string(mesh.vertex_normals.reshape(-1))[1:-1]
    replacement['VCOUNT'] = str(len(mesh.vertices))
    replacement['VCOUNTX3'] = str(len(mesh.vertices) * 3)
    replacement['FCOUNT'] = str(len(mesh.faces))

    # 使用模板替换字符串
    export = template.substitute(replacement)
    return export


def export_dict64(mesh):
    # 使用 base64 编码导出网格为字典格式
    return export_dict(mesh, encoding='base64')


def export_dict(mesh, encoding=None):
    # 将网格导出为字典格式,支持不同的编码
    def encode(item, dtype=None):
        if encoding is None:
            return item.tolist()
        else:
            return array_to_encoded(item, dtype=dtype, encoding=encoding)

    export = {'metadata': tolist_dict(mesh.metadata),
              'faces': encode(mesh.faces, np.uint32),
              'face_normals': encode(mesh.face_normals, np.float32),
              'vertices': encode(mesh.vertices, np.float32)}
    return export


def export_json(mesh):
    # 将网格导出为 JSON 格式
    blob = export_dict(mesh, encoding='base64')
    export = json.dumps(blob)
    return export


def export_msgpack(mesh):
    # 将网格导出为 MsgPack 格式
    import msgpack
    blob = export_dict(mesh, encoding='binary')
    export = msgpack.dumps(blob)
    return export


_mesh_exporters = {'ply': export_ply,
                   'stl': export_stl,
                   'dict': export_dict,
                   'json': export_json,
                   'off': export_off,
                   'dae': export_collada,
                   'dict64': export_dict64,
                   'msgpack': export_msgpack,
                   'collada': export_collada}
