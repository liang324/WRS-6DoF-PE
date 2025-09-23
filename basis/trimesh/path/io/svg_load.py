import numpy as np
from ...constants import log
from ..entities import Line, Arc, Bezier
from collections import deque
from xml.dom.minidom import parseString as parse_xml

try:
    from svg.path import parse_path
except:
    log.warning('SVG path loading unavailable!')


def svg_to_path(file_obj, file_type=None):
    '''
    将 SVG 文件转换为路径对象

    :param file_obj: 文件对象,包含 SVG 数据
    :param file_type: 文件类型(可选)
    :return: 包含实体和顶点的字典
    '''

    def complex_to_float(values):
        '''
        将复数值转换为浮点数组

        :param values: 复数值列表
        :return: 浮点数组
        '''
        return np.array([[i.real, i.imag] for i in values])

    def load_line(svg_line):
        '''
        加载 SVG 线段并转换为路径实体

        :param svg_line: SVG 线段对象
        '''
        points = complex_to_float([svg_line.point(0.0), svg_line.point(1.0)])
        if not starting: points[0] = vertices[-1]
        entities.append(Line(np.arange(2) + len(vertices)))
        vertices.extend(points)

    def load_arc(svg_arc):
        '''
        加载 SVG 弧线并转换为路径实体

        :param svg_arc: SVG 弧线对象
        '''
        points = complex_to_float([svg_arc.start,
                                   svg_arc.point(.5),
                                   svg_arc.end])
        if not starting: points[0] = vertices[-1]
        entities.append(Arc(np.arange(3) + len(vertices)))
        vertices.extend(points)

    def load_quadratic(svg_quadratic):
        '''
        加载 SVG 二次贝塞尔曲线并转换为路径实体

        :param svg_quadratic: SVG 二次贝塞尔曲线对象
        '''
        points = complex_to_float([svg_quadratic.start,
                                   svg_quadratic.control,
                                   svg_quadratic.end])
        if not starting: points[0] = vertices[-1]
        entities.append(Bezier(np.arange(3) + len(vertices)))
        vertices.extend(points)

    def load_cubic(svg_cubic):
        '''
        加载 SVG 三次贝塞尔曲线并转换为路径实体

        :param svg_cubic: SVG 三次贝塞尔曲线对象
        '''
        points = complex_to_float([svg_cubic.start,
                                   svg_cubic.control1,
                                   svg_cubic.control2,
                                   svg_cubic.end])
        if not starting: points[0] = vertices[-1]
        entities.append(Bezier(np.arange(4) + len(vertices)))
        vertices.extend(points)

    # 从 XML 文件中提取所有路径字符串
    xml = parse_xml(file_obj.read())
    paths = [p.attributes['d'].value for p in xml.getElementsByTagName('path')]
    entities = deque()
    vertices = deque()
    loaders = {'Arc': load_arc,
               'Line': load_line,
               'CubicBezier': load_cubic,
               'QuadraticBezier': load_quadratic}
    for svg_string in paths:
        starting = True
        for svg_entity in parse_path(svg_string):
            loaders[svg_entity.__class__.__name__](svg_entity)

    return {'entities': np.array(entities),
            'vertices': np.array(vertices)}
