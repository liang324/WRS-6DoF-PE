import numpy as np
from collections import OrderedDict
from string import Template
from ..templates import get_template


def load_ply(file_obj, *args, **kwargs):
    '''
    从打开的文件对象加载 PLY 文件

    :param file_obj: 打开的文件对象,用于读取 PLY 数据
    :param args: 其他位置参数
    :param kwargs: 其他关键字参数
    :return: dict,包含网格信息的字典,可以传递给 Trimesh 构造函数,例如:a = Trimesh(**mesh_kwargs)
    '''
    # 从头部填充的有序字典
    elements = ply_read_header(file_obj)
    # 某些元素传递时,列表维度未包含在头部中,因此此函数进入文件的主要部分并获取列表维度,然后作为单个操作读取主要数据
    ply_populate_listsize(file_obj, elements)
    # 文件中剩余的字节数
    size_file = size_to_end(file_obj)
    # 头部描述的数据结构应占用的字节数
    size_elements = ply_elements_size(elements)
    # 如果字节数不相同,则文件可能已损坏
    if size_file != size_elements:
        raise ValueError('File is unexpected length!')
    # 当所有内容都被填充并且有合理的可信度时,文件是完整的,读取头中描述的数据字段
    ply_populate_data(file_obj, elements)
    # 所有数据现在都存储在元素中,但我们需要将其作为一组关键字参数传递给 Trimesh 构造函数
    # 结果类似于 {'vertices' : (数据), 'faces' : (数据)}
    mesh_kwargs = ply_elements_kwargs(elements)
    return mesh_kwargs


def export_ply(mesh):
    '''
    以 PLY 格式导出网格

    :param mesh: Trimesh 对象
    :return: bytes,导出结果的字节数据
    '''
    dtype_face = np.dtype([('count', '<u1'),
                           ('index', '<i4', (3))])
    dtype_vertex = np.dtype([('vertex', '<f4', (3))])
    faces = np.zeros(len(mesh.faces), dtype=dtype_face)
    faces['count'] = 3
    faces['index'] = mesh.faces
    vertex = np.zeros(len(mesh.vertices), dtype=dtype_vertex)
    vertex['vertex'] = mesh.vertices
    template = Template(get_template('ply.template'))
    export = template.substitute({'vertex_count': len(mesh.vertices), 'face_count': len(mesh.faces)}).encode('utf-8')
    export += vertex.tostring()
    export += faces.tostring()
    return export


def ply_element_colors(element):
    '''
    给定一个元素,尝试从其属性中提取 RGBA 颜色,并将其作为 (n,3|4) 数组返回

    :param element: 包含属性的元素
    :return: 如果找到颜色,返回颜色数组；否则返回 None
    '''
    color_keys = ['red', 'green', 'blue', 'alpha']
    candidate_colors = [element['log_filename'][i] for i in color_keys if i in element['properties']]
    if len(candidate_colors) >= 3:
        return np.column_stack(candidate_colors)
    return None


def ply_read_header(file_obj):
    '''
    读取 PLY 文件的 ASCII 头部,并将文件对象定位到数据的起始位置

    :param file_obj: 打开的文件对象,用于读取 PLY 数据
    :return: OrderedDict,包含元素及其属性的信息
    '''
    # 从 PLY 规范中定义的数据类型
    dtypes = {'char': 'i1',
              'uchar': 'u1',
              'short': 'i2',
              'ushort': 'u2',
              'int': 'i4',
              'uint': 'u4',
              'float': 'f4',
              'double': 'f8'}
    # 检查文件是否为 PLY 格式
    if not 'ply' in str(file_obj.readline()):
        raise ValueError('这不是一个 PLY 文件')
    encoding = str(file_obj.readline()).strip().split()[1]

    # 不支持 ASCII 编码的 PLY 文件
    if 'ascii' in encoding:
        raise ValueError('不支持 ASCII 编码的 PLY 文件！')

    # 确定字节序
    endian = ['<', '>']['big' in encoding]
    elements = OrderedDict()

    while True:
        line = file_obj.readline()
        if line is None:
            raise ValueError('Header wasn\'t terminated properly!')
        line = line.decode('utf-8').strip().split()

        if 'end_header' in line:
            break

        if 'element' in line[0]:
            name, length = line[1:]
            elements[name] = {'length': int(length), 'properties': OrderedDict()}

        elif 'property' in line[0]:
            if len(line) == 3:
                dtype, field = line[1:]
                elements[name]['properties'][str(field)] = endian + dtypes[dtype]
            elif 'list' in line[1]:
                dtype_count, dtype, field = line[2:]
                elements[name]['properties'][str(field)] = (endian +
                                                            dtypes[dtype_count] +
                                                            ', ($LIST,)' +
                                                            endian +
                                                            dtypes[dtype])
    return elements


def ply_populate_listsize(file_obj, elements):
    '''
    给定从头部填充的元素集合,如果有任何列表属性,则在文件中查找列表的长度

    :param file_obj: 打开的文件对象,用于读取 PLY 数据
    :param elements: OrderedDict,包含元素及其属性的信息
    '''
    p_start = file_obj.tell()
    p_current = file_obj.tell()
    for element_key, element in elements.items():
        props = element['properties']
        prior_data = ''
        for k, dtype in props.items():
            if '$LIST' in dtype:
                # 每个列表字段有两种数据类型:列表长度(单个值)和列表数据(多个值)这里只读取列表长度的单个值
                field_dtype = np.dtype(dtype.split(',')[0])
                if len(prior_data) == 0:
                    offset = 0
                else:
                    offset = np.dtype(prior_data).itemsize()
                file_obj.seek(p_current + offset)
                size = np.fromstring(file_obj.read(field_dtype.itemsize), dtype=field_dtype)[0]
                props[k] = props[k].replace('$LIST', str(size))
            prior_data += props[k] + ','
        itemsize = np.dtype(', '.join(props.values())).itemsize
        p_current += element['length'] * itemsize
    file_obj.seek(p_start)


def ply_populate_data(file_obj, elements):
    '''
    根据头部的信息读取数据,并将其添加到元素的 'log_filename' 字段中

    :param file_obj: 打开的文件对象,用于读取 PLY 数据
    :param elements: OrderedDict,包含元素及其属性的信息
    :return: OrderedDict,包含填充数据的元素
    '''
    for key in elements.keys():
        items = list(elements[key]['properties'].items())
        dtype = np.dtype(items)
        data = file_obj.read(elements[key]['length'] * dtype.itemsize)
        elements[key]['log_filename'] = np.fromstring(data, dtype=dtype)
    return elements


def ply_elements_kwargs(elements):
    '''
    从元素数据结构中提取 Trimesh 对象构造函数所需的关键字参数

    :param elements: OrderedDict,包含元素及其属性的信息
    :return: dict,包含网格信息的字典
    '''
    vertices = np.column_stack([elements['vertex']['log_filename'][i] for i in 'xyz'])
    faces = elements['face']['log_filename']['vertex_indices']['f1']
    face_colors = ply_element_colors(elements['face'])
    vertex_colors = ply_element_colors(elements['vertex'])
    result = {'vertices': vertices,
              'faces': faces,
              'face_colors': face_colors,
              'vertex_colors': vertex_colors}
    return result


def ply_elements_size(elements):
    '''
    给定从头部填充的elements数据结构,计算文件完整时应该有多长

    :param elements: OrderedDict,包含元素及其属性的信息
    :return: int,文件的预期字节长度
    '''
    size = 0
    for element in elements.values():
        dtype = np.dtype(','.join(element['properties'].values()))
        size += element['length'] * dtype.itemsize
    return size


def size_to_end(file_obj):
    '''
    给定一个打开的文件对象,返回到文件末尾的字节数

    :param file_obj: 打开的文件对象
    :return: int,到文件末尾的字节数
    '''
    position_current = file_obj.tell()
    file_obj.seek(0, 2)
    position_end = file_obj.tell()
    file_obj.seek(position_current)
    size = position_end - position_current
    return size


_ply_loaders = {'ply': load_ply}
