from .interfaces import scad, blender


def difference(meshes, engine=None):
    '''
    计算一个网格与其他多个网格的布尔差集

    :param meshes: Trimesh 对象的列表,表示参与运算的网格
    :param engine: 字符串,指定使用哪个后端.有效选项为 'blender' 或 'scad'
    :return: Trimesh 对象,表示 a - (其他网格) 的结果
    '''
    result = _engines[engine](meshes, operation='difference')
    return result


def union(meshes, engine=None):
    '''
    计算一个网格与其他多个网格的布尔并集

    :param meshes: Trimesh 对象的列表,表示参与运算的网格
    :param engine: 字符串,指定使用哪个后端.有效选项为 'blender' 或 'scad'
    :return: Trimesh 对象,表示 a + (其他网格) 的结果
    '''
    result = _engines[engine](meshes, operation='union')
    return result


def intersection(meshes, engine=None):
    '''
    计算一个网格与其他多个网格的布尔交集

    :param meshes: Trimesh 对象的列表,表示参与运算的网格
    :param engine: 字符串,指定使用哪个后端.有效选项为 'blender' 或 'scad'
    :return: Trimesh 对象,表示所有网格共同包含的体积
    '''
    result = _engines[engine](meshes, operation='intersection')
    return result


def boolean_automatic(meshes, operation):
    '''
    自动选择可用的后端来执行布尔运算

    :param meshes: Trimesh 对象的列表,表示参与运算的网格
    :param operation: 字符串,指定要执行的布尔运算类型(如 'union', 'difference', 'intersection')
    :return: Trimesh 对象,表示布尔运算的结果
    '''
    if blender.exists:
        result = blender.boolean(meshes, operation)
    elif scad.exists:
        result = scad.boolean(meshes, operation)
    else:
        raise ValueError('没有可用的后端来执行布尔运算！')
    return result


_engines = {None: boolean_automatic,
            'auto': boolean_automatic,
            'scad': scad.boolean,
            'blender': blender.boolean}
