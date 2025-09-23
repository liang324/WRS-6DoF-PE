import numpy as np
from ..constants import log


def load_assimp(file_obj, file_type=None):
    '''
    使用 assimp 库从文件对象和类型或文件名(如果file_obj是字符串)加载网格

    Assimp支持大量的网格格式
    性能注意:在二进制STL测试中,pyassimp比此包中包含的本地加载器慢约10倍
    这可能是由于他们对数据结构的递归美化处理
    此外,您需要一个非常新的PyAssimp版本才能使此函数正常工作 大约在2014年9月5日合并到 assimp github主分支的提交

    :param file_obj: 文件对象或文件名字符串
    :param file_type: 文件类型,如果未提供,将从文件名推断
    :return: 包含网格数据的字典或字典列表
    '''

    def LPMesh_to_Trimesh(lp):
        # 将颜色数据转换为整数并调整形状
        colors = (np.reshape(lp.colors, (-1, 4))[:, 0:3] * 255).astype(int)
        return {'vertices': lp.vertices,
                'vertex_normals': lp.normals,
                'faces': lp.faces,
                'vertex_colors': colors}

    if not hasattr(file_obj, 'read'):
        # 如果没有read属性,我们假设传入的是文件名
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'rb')

    # 使用pyassimp加载场景
    scene = pyassimp.load(file_obj, file_type=file_type)
    # 将场景中的网格转换为Trimesh格式
    meshes = list(map(LPMesh_to_Trimesh, scene.meshes))
    pyassimp.release(scene)

    if len(meshes) == 1:
        return meshes[0]
    return meshes


_assimp_loaders = {}
try:
    import pyassimp

    if hasattr(pyassimp, 'available_formats'):
        # 获取可用格式列表
        _assimp_formats = [i.lower() for i in pyassimp.available_formats()]
    else:
        log.warning('检测到旧版本的assimp,使用硬编码格式列表')
        _assimp_formats = ['dae', 'blend', '3ds', 'ase', 'obj',
                           'ifc', 'xgl', 'zgl', 'ply', 'lwo',
                           'lxo', 'x', 'ac', 'ms3d', 'cob', 'scn']
    # 更新加载器字典
    _assimp_loaders.update(zip(_assimp_formats,
                               [load_assimp] * len(_assimp_formats)))
except ImportError:
    log.warning('Pyassimp不可用,仅使用本地加载器')
