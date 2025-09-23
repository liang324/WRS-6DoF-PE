import numpy as np
import networkx as nx
import itertools
from collections import deque
from tempfile import NamedTemporaryFile
from distutils.spawn import find_executable
from subprocess import check_call
from xml.etree import cElementTree
from ..constants import res, log

_METERS_TO_INCHES = 1.0 / .0254
_STEP_FACETER = find_executable('export_product_asm')


def load_step(file_obj, file_type=None):
    '''
    使用 STEPtools Inc. Author Tools 二进制工具对 STEP 文件进行网格化,并返回 Trimesh 对象的列表

    使用此工具而非 openCASCADE,因为它显著更稳定(尽管不是开源软件)

    STEPtools Inc. 提供的二进制工具的许可证:
    http://www.steptools.com/demos/license_author.html

    要将所需的二进制文件('export_product_asm')安装到 PATH 中:
         wget http://www.steptools.com/demos/stpidx_author_linux_x86_64_16.0.zip
        unzip stpidx_author_linux_x86_64_16.0.zip
        sudo cp stpidx_author_linux_x86_64/bin/export_product_asm /usr/bin/

    :param file_obj: 类文件对象,包含 STEP 文件数据
    :param file_type: 未使用的参数
    :return: 包含 Trimesh 对象的列表(具有从 STEP 文件中设置的正确元数据)
    '''
    # 使用临时文件存储输出
    with NamedTemporaryFile() as out_file:
        with NamedTemporaryFile(suffix='.STEP') as in_file:
            # 检查 file_obj 是否具有 'read' 方法
            if hasattr(file_obj, 'read'):
                in_file.write(file_obj.read())
                in_file.seek(0)
                file_name = in_file.name
            else:
                file_name = file_obj
            # 调用外部程序进行网格化处理
            check_call([_STEP_FACETER, file_name,
                        '-tol', str(res.mesh),
                        '-o', out_file.name])
            # 解析输出 XML 文件
            t = cElementTree.parse(out_file)
    meshes = {}
    # 从 XML 文档中获取没有元数据的网格
    for shell in t.findall('shell'):
        # 查询 XML 结构以获取顶点和面
        vertices = np.array([v.get('p').split() for v in shell.findall('.//v')], dtype=float)
        faces = np.array([f.get('v').split() for f in shell.findall('.//f')], dtype=int)
        # 法线不总是返回,但面具有正确的绕线,因此可以通过点积正确生成
        mesh = {'vertices': vertices,
                'faces': faces,
                'metadata': {}}
        # 通过 ID 引用存储网格
        meshes[shell.get('id')] = mesh

    try:
        # 填充形状和变换的图
        g = nx.MultiDiGraph()
        # 键:{网格 ID : 形状 ID}
        mesh_shape = {}
        # 假设文档具有一致的单位
        to_inches = None
        for shape in t.findall('shape'):
            shape_id = shape.get('id')
            shape_unit = shape.get('unit')
            mesh_id = shape.get('shell')
            if not shape_unit is None:
                to_inches = float(shape_unit.split()[1]) * _METERS_TO_INCHES
            if not mesh_id is None:
                for i in mesh_id.split():
                    mesh_shape[i] = shape_id
                # g.node[shape_id]['mesh'] = mesh_id
                g.add_node(shape_id, {'mesh': mesh_id})

            for child in shape.getchildren():
                child_id = child.get('ref')
                transform = np.array(child.get('xform').split(), dtype=float).reshape((4, 4)).T
                g.add_edge(shape_id, child_id, transform=transform)

        # 哪个产品 ID 具有根形状
        prod_root = t.getroot().get('root')
        shape_root = None
        for prod in t.findall('product'):
            prod_id = prod.get('id')
            prod_name = prod.get('name')
            prod_shape = prod.get('shape')
            if prod_id == prod_root:
                shape_root = prod_shape
            g.node[prod_shape]['product_name'] = prod_name

        # 现在装配树已经填充,遍历它以找到我们提取的网格的最终变换和数量
        for mesh_id in meshes.keys():
            shape_id = mesh_shape[mesh_id]
            transforms_all = deque()
            path_str = deque()
            if shape_id == shape_root:
                paths = [[shape_id, shape_id]]
            else:
                paths = nx.all_simple_paths(g, shape_root, shape_id)
            paths = np.array(list(paths))
            garbage, unique = np.unique(['.'.join(i) for i in paths], return_index=True)
            paths = paths[unique]
            for path in paths:
                path_name = [g.node[i]['product_name'] for i in path[:-1]]
                edges = np.column_stack((path[:-1],
                                         path[:-1])).reshape(-1)[1:-1].reshape((-1, 2))
                transforms = [np.eye(4)]
                for e in edges:
                    # 从边缘获取每个变换
                    local = [i['transform'] for i in g.edge[e[0]][e[1]].values()]
                    # 所有变换都是顺序的,因此我们需要组合
                    transforms = [np.dot(*i) for i in itertools.product(transforms, local)]
                transforms_all.extend(transforms)
                path_str.extend(['/'.join(path_name)] * len(transforms))
            meshes[mesh_id]['vertices'] *= to_inches
            meshes[mesh_id]['metadata']['units'] = 'inches'
            meshes[mesh_id]['metadata']['name'] = path_name[-1]
            meshes[mesh_id]['metadata']['paths'] = np.array(path_str)
            meshes[mesh_id]['metadata']['quantity'] = len(transforms_all)
            meshes[mesh_id]['metadata']['transforms'] = np.array(transforms_all)
    except:
        log.error('STEP 加载处理错误,中止元数据！', exc_info=True)
    return meshes.values()


if _STEP_FACETER is None:
    log.debug('STEP 加载不可用！')
    _step_loaders = {}
else:
    _step_loaders = {'step': load_step, 'stp': load_step}
