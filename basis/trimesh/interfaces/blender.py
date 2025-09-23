from .generic import MeshScript
from ..templates import get_template
from distutils.spawn import find_executable

_blender_executable = find_executable('blender')
_blender_template = get_template('blender.py.template')
exists = _blender_executable is not None


def boolean(meshes, operation='difference'):
    """
    执行布尔操作以合并或修改网格

    :param meshes: 要进行布尔操作的网格列表
    :param operation: 布尔操作的类型,可以是 'difference'(差集)、'union'(并集)或 'intersection'(交集)
                      默认值为 'difference'
    :return: 包含布尔操作结果的字典
    :raises ValueError: 如果Blender不可用,则抛出此异常
    """
    if not exists:
        raise ValueError('No blender available!')
    operation = str.upper(operation)
    if operation == 'INTERSECTION':
        operation = 'INTERSECT'
    script = _blender_template.replace('$operation', operation)
    with MeshScript(meshes=meshes, script=script) as blend:
        result = blend.run(_blender_executable + ' --background --python $script')
    # Blender返回的面法线可能不正确
    result['face_normals'] = None
    return result
