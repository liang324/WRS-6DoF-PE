import numpy as np
from ...constants import log
from ...constants import tol_path as tol
from ..entities import Line, Arc, BSpline
from ..util import angles_to_threepoint, is_ccw
from ...util import is_binary_file, multi_dict
from collections import deque

#单元代码
_DXF_UNITS = {1: 'inches',
              2: 'feet',
              3: 'miles',
              4: 'millimeters',
              5: 'centimeters',
              6: 'meters',
              7: 'kilometers',
              8: 'microinches',
              9: 'mils',
              10: 'yards',
              11: 'angstroms',
              12: 'nanometers',
              13: 'microns',
              14: 'decimeters',
              15: 'decameters',
              16: 'hectometers',
              17: 'gigameters',
              18: 'AU',
              19: 'light years',
              20: 'parsecs'}


def get_key(blob, field, code):
    """
    从给定的 DXF 数据块中提取指定字段的值

    :param blob: 包含 DXF 数据的二维数组,每行包含组代码和对应的值
    :param field: 要查找的字段名称(字符串)
    :param code: 期望的组代码(字符串),用于验证字段值
    :return: 如果找到匹配的字段和组代码,返回字段值(整数)；否则返回 None
    """
    try:
        line = blob[np.nonzero(blob[:, 1] == field)[0][0] + 1]
    except IndexError:
        return None
    if line[0] == code:
        return int(line[1])
    else:
        return None


def load_dxf(file_obj):
    '''
    加载 DXF 文件并返回包含顶点和实体信息的字典

    :param file_obj: 文件对象或类似文件的对象(具有 read 方法)
    :return: 包含实体、顶点和元数据的字典
    '''

    def convert_line(e_data):
        # 将线段实体转换为顶点和实体
        e = dict(e_data)
        entities.append(Line(len(vertices) + np.arange(2)))
        vertices.extend(np.array([[e['10'], e['20']], [e['11'], e['21']]], dtype=float))

    def convert_circle(e_data):
        # 将圆形实体转换为顶点和实体
        e = dict(e_data)
        R = float(e['40'])
        C = np.array([e['10'], e['20']]).astype(float)
        points = angles_to_threepoint([0, np.pi], C[0:2], R)
        entities.append(Arc(points=(len(vertices) + np.arange(3)), closed=True))
        vertices.extend(points)

    def convert_arc(e_data):
        # 将弧形实体转换为顶点和实体
        e = dict(e_data)
        R = float(e['40'])
        C = np.array([e['10'], e['20']], dtype=float)
        A = np.radians(np.array([e['50'], e['51']], dtype=float))
        points = angles_to_threepoint(A, C[0:2], R)
        entities.append(Arc(len(vertices) + np.arange(3), closed=False))
        vertices.extend(points)

    def convert_polyline(e_data):
        # 将多段线实体转换为顶点和实体
        e = multi_dict(e_data)
        lines = np.column_stack((e['10'], e['20'])).astype(float)
        entities.append(Line(np.arange(len(lines)) + len(vertices)))
        vertices.extend(lines)

    def convert_bspline(e_data):
        # 在DXF中有n个点和n个有序字段具有相同的分组代码
        e = multi_dict(e_data)
        points = np.column_stack((e['10'], e['20'])).astype(np.float)
        knots = np.array(e['40']).astype(float)
        #检查欧氏距离是否闭合
        closed = np.linalg.norm(points[0] - points[-1]) < tol.merge
        #如果它是闭合的,确保它是CCW,用于后面的多边形幸福度
        if closed and (not is_ccw(np.vstack((points, points[0])))):
            points = points[::-1]
        entities.append(BSpline(points=np.arange(len(points)) + len(vertices),
                                knots=knots,
                                closed=closed))
        vertices.extend(points)

    if is_binary_file(file_obj):
        raise ValueError("不支持二进制 DXF 文件！")

    # 在DXF文件中,行是成对的
    # 分组代码,下一行是值
    # 我们移除所有空格,然后用
    # splitlines函数,使用通用的newline方法
    raw = str(file_obj.read().decode('ascii').upper().replace(' ', ''))
    # 如果重塑失败,则意味着DXF格式错误
    blob = np.array(str.splitlines(raw)).reshape((-1, 2))

    # 获取 DXF 文件中包含头部的部分
    endsec = np.nonzero(blob[:, 1] == 'ENDSEC')[0]
    header_start = np.nonzero(blob[:, 1] == 'HEADER')[0][0]
    header_end = endsec[np.searchsorted(endsec, header_start)]
    header_blob = blob[header_start:header_end]

    # 获取 DXF 文件中包含实体的部分
    entity_start = np.nonzero(blob[:, 1] == 'ENTITIES')[0][0]
    entity_end = endsec[np.searchsorted(endsec, entity_start)]
    entity_blob = blob[entity_start:entity_end]

    # 从 DXF 的头部提取元数据
    metadata = dict()
    units = get_key(header_blob, '$INSUNITS', '70')
    if units in _DXF_UNITS:
        metadata['units'] = _DXF_UNITS[units]
    else:
        log.warning('DXF 文件未指定单位！')

    # 找到实体的起始点
    # group_check = np.logical_or(entity_blob[:,0] == '0',
    #                            entity_blob[:,0] == '5')
    group_check = entity_blob[:, 0] == '0'
    inflection = np.nonzero(group_check)[0]

    # inflection = np.nonzero(np.logical_and(group_check[:-1],
    #                                       group_check[:-1] == group_check[1:]))[0]
    loaders = {'LINE': convert_line,
               'LWPOLYLINE': convert_polyline,
               'ARC': convert_arc,
               'CIRCLE': convert_circle,
               'SPLINE': convert_bspline}
    vertices = deque()
    entities = deque()

    for chunk in np.array_split(entity_blob, inflection):
        if len(chunk) > 2:
            entity_type = chunk[0][1]
            if entity_type in loaders:
                loaders[entity_type](chunk)
            else:
                log.debug('Entity type %s not supported', entity_type)

    result = {'vertices': np.vstack(vertices).astype(np.float64),
              'entities': np.array(entities),
              'metadata': metadata}

    return result
