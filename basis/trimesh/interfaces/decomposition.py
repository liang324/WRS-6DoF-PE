import os
import platform
from basis.trimesh.interfaces.generic import MeshScript
from basis.trimesh.constants import log
from distutils.spawn import find_executable

_search_path = os.environ['PATH']
if platform.system() == 'Windows':
    # 用分隔符分割现有路径
    _search_path = [i for i in _search_path.split(';') if len(i) > 0]
    _search_path.append(r'C:\Program Files')
    _search_path.append(r'C:\Program Files (x86)')
    _search_path = ';'.join(_search_path)
    log.debug('searching for vhacd in: %s', _search_path)

_vhacd_executable = None
for _name in ['vhacd', 'testVHACD']:
    _vhacd_executable = find_executable(_name, path=_search_path)
    if _vhacd_executable is not None:
        break

exists = _vhacd_executable is not None


def convex_decomposition(mesh, debug=False, **kwargs):
    """
    使用 VHACD 生成单个网格的近似凸分解

    :param mesh: 要分解为凸组件的网格
    :param debug: 是否启用调试模式
    :param kwargs: 其他可选参数
    :return: (n,) trimesh.Trimesh 凸网格的列表
    """
    if not exists:
        raise ValueError('没有可用的 vhacd！')
    argstring = ' --input $MESH_0 --output $MESH_POST --log $SCRIPT'
    # 从输入字典中传递额外的参数
    for key, value in kwargs.items():
        argstring += ' --{} {}'.format(str(key), str(value))
    with MeshScript(meshes=[mesh],
                    script='',
                    exchange='obj',
                    group_material=False,
                    split_object=True,
                    debug=debug) as vhacd:
        result = vhacd.run(_vhacd_executable + argstring)
    # 如果返回的是一个场景,则返回网格列表
    if hasattr(result, 'geometry') and isinstance(result.geometry, dict):
        return list(result.geometry.values())
    return result
