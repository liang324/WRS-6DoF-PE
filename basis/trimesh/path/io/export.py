import numpy as np
import json
from string import Template
from ..arc import arc_center
from ...templates import get_template
from ...util import three_dimensionalize
from ...constants import log
from ...constants import res_path as res

_templates_dxf = {k: Template(v) for k, v in json.loads(get_template('dxf.json.template')).items()}


def export_path(path, file_type, file_obj=None):
    '''
    将 Path 对象导出到文件对象或文件名

    :param path: 要导出的 Path 对象
    :param file_type: 字符串,表示文件类型(例如:'svg')
    :param file_obj: 文件名字符串或文件对象
    :return: 导出的文件内容
    '''
    # 检查 file_obj 是否为文件对象,如果不是则打开文件
    if ((not hasattr(file_obj, 'read')) and
            (not file_obj is None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'wb')
    # 使用对应的导出器导出路径
    export = _path_exporters[file_type](path)
    # 写入导出的内容
    return _write_export(export, file_obj)


def export_dict(path):
    '''
    将 Path 对象导出为字典格式

    :param path: 要导出的 Path 对象
    :return: 包含实体和顶点信息的字典
    '''
    # 将每个实体转换为字典
    export_entities = [e.to_dict() for e in path.entities]
    # 创建导出对象
    export_object = {'entities': export_entities,
                     'vertices': path.vertices.tolist()}
    return export_object


def export_svg(drawing):
    '''
    将路径绘图转换为 SVG 路径字符串

    'holes' 将以相反的顺序呈现,以便渲染库可以将其渲染为孔
    :param drawing: 包含路径信息的绘图对象
    :return: SVG 路径字符串
    '''
    points = drawing.vertices.view(np.ndarray)

    # 定义用于转换圆形和弧形的辅助函数
    def circle_to_svgpath(center, radius, reverse):
        radius_str = format(radius, res.export)
        path_str = '  M' + format(center[0] - radius, res.export) + ','
        path_str += format(center[1], res.export)
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(2 * radius, res.export) + ',0'
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(-2 * radius, res.export) + ',0Z  '
        return path_str

    def svg_arc(arc, reverse):
        '''
        圆弧字符串:(rx x-axis-rotation large-arc-flag sweep-flag x y)+
        大弧度:大于180度
        扫描标志:方向(cw/ccw)

        :param arc: 弧形对象
        :param reverse: 布尔值,是否反向
        :return: SVG 路径字符串
        '''
        vertices = points[arc.points[::((reverse * -2) + 1)]]
        vertex_start, vertex_mid, vertex_end = vertices
        center_info = arc_center(vertices)
        C, R, N, angle = (center_info['center'],
                          center_info['radius'],
                          center_info['normal'],
                          center_info['span'])
        if arc.closed: return circle_to_svgpath(C, R, reverse)
        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(vertex_mid - vertex_start, vertex_end - vertex_start) > 0))

        R_ex = format(R, res.export)
        x_ex = format(vertex_end[0], res.export)
        y_ex = format(vertex_end[1], res.export)
        arc_str = 'A' + R_ex + ',' + R_ex + ' 0 '
        arc_str += large_flag + ',' + sweep_flag + ' '
        arc_str += x_ex + ',' + y_ex
        return arc_str

    def svg_line(line, reverse):
        # 将线段转换为 SVG 路径字符串
        vertex_end = points[line.points[-(not reverse)]]
        x_ex = format(vertex_end[0], res.export)
        y_ex = format(vertex_end[1], res.export)
        line_str = 'L' + x_ex + ',' + y_ex
        return line_str

    def svg_moveto(vertex_id):
        # 将移动命令转换为 SVG 路径字符串
        x_ex = format(points[vertex_id][0], res.export)
        y_ex = format(points[vertex_id][1], res.export)
        move_str = 'M' + x_ex + ',' + y_ex
        return move_str

    def convert_path(path, reverse=False):
        # 转换路径为 SVG 路径字符串
        path = path[::(reverse * -2) + 1]
        path_str = svg_moveto(drawing.entities[path[0]].end_points[-reverse])
        for i, entity_id in enumerate(path):
            entity = drawing.entities[entity_id]
            e_type = entity.__class__.__name__
            try:
                path_str += converters[e_type](entity, reverse)
            except KeyError:
                log.warn('%s entity not available for export!', e_type)
        path_str += 'Z'
        return path_str

    # 定义转换器字典
    converters = {'Line': svg_line, 'Arc': svg_arc}
    path_str = ''
    for path_index, path in enumerate(drawing.paths):
        reverse = not (path_index in drawing.workdir)
        path_str += convert_path(path, reverse)
    return path_str


def export_dxf(path):
    '''
    将 Path 对象导出为 DXF 格式的字符串

    :param path: 要导出的 Path 对象
    :return: 包含 DXF 数据的字符串
    '''

    # 定义辅助函数以格式化点坐标
    def format_points(points, increment=True):
        points = np.asanyarray(points)
        three = three_dimensionalize(points, return_2D=False)
        if increment:
            group = np.tile(np.arange(len(three)).reshape((-1, 1)), (1, 3))
        else:
            group = np.zeros((len(three), 3), dtype=int)
        group += [10, 20, 30]
        interleaved = np.dstack((group.astype(str),
                                 three.astype(str))).reshape(-1)
        packed = '\n'.join(interleaved)
        return packed

    def entity_color_layer(entity):
        # 获取实体的颜色和图层信息
        color, layer = 0, 0
        if hasattr(entity, 'color'): color = int(entity.color)
        if hasattr(entity, 'layer'): layer = int(entity.layer)
        return color, layer

    def convert_line(line, vertices):
        # 将线段转换为 DXF 格式
        points = line.discrete(vertices)
        color, layer = entity_color_layer(line)
        is_poly = len(points) > 2
        line_type = ['LINE', 'LWPOLYLINE'][int(is_poly)]
        result = templates['line'].substitute({'TYPE': line_type,
                                               'POINTS': format_points(points, increment=not is_poly),
                                               'NAME': str(id(line))[:16],
                                               'LAYER_NUMBER': layer,
                                               'COLOR_NUMBER': color})
        return result

    def convert_arc(arc, vertices):
        # 将弧形转换为 DXF 格式
        info = arc.center(vertices)
        color, layer = entity_color_layer(arc)
        angles = np.degrees(info['angles'])
        arc_type = ['ARC', 'CIRCLE'][int(arc.closed)]
        result = templates['arc'].substitute({'TYPE': arc_type,
                                              'CENTER_POINT': format_points([info['center']]),
                                              'ANGLE_MIN': angles[0],
                                              'ANGLE_MAX': angles[1],
                                              'RADIUS': info['radius'],
                                              'LAYER_NUMBER': layer,
                                              'COLOR_NUMBER': color})
        return result

    def convert_generic(entity, vertices):
        # 通用实体转换为 DXF 格式
        log.warning('没有可用的转换器！导出为折线！')
        return convert_line(entity, vertices)

    templates = _templates_dxf
    np.set_printoptions(precision=12)
    conversions = {'Line': convert_line,
                   'Arc': convert_arc,
                   'Bezier': convert_generic,
                   'BSpline': convert_generic}
    entities_str = ''
    for e in path.entities:
        name = type(e).__name__
        if name in conversions:
            entities_str += conversions[name](e, path.vertices)
        else:
            log.warning('Entity type %s not exported!', name)

    header = templates['header'].substitute({'BOUNDS_MIN': format_points([path.bounds[0]]),
                                             'BOUNDS_MAX': format_points([path.bounds[1]]),
                                             'UNITS_CODE': '1'})
    entities = templates['entities'].substitute({'ENTITIES': entities_str})
    footer = templates['footer'].substitute()
    result = '\n'.join([header, entities, footer])
    return result


def _write_export(export, file_obj=None):
    '''
    将字符串写入文件,如果未指定 file_obj,则返回字符串

    :param export: 导出数据的字符串
    :param file_obj: 文件对象或文件名
    :return: 导出的字符串
    '''

    if file_obj is None:
        return export
    elif hasattr(file_obj, 'write'):
        out_file = file_obj
    else:
        out_file = open(file_obj, 'wb')
    out_file.write(export)
    out_file.close()
    return export


_path_exporters = {'dxf': export_dxf,
                   'svg': export_svg,
                   'dict': export_dict}
