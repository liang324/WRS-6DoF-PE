from .generic import MeshScript
from distutils.spawn import find_executable

_scad_executable = find_executable('openscad')
exists = _scad_executable is not None


def interface_scad(meshes, script):
    '''
    与OpenSCAD进行交互的方式,OpenSCAD本身是CGAL CSG绑定的
    CGAL非常稳定,但安装和使用可能比较困难,因此此函数提供了一种使用临时文件的解决方案,以获得基本的 CGAL CSG 功能

    :param meshes: 要处理的Trimesh对象列表
    :param script: 要发送到OpenSCAD的脚本字符串
    :return: 执行结果,通常是处理后的网格数据
    '''
    if not exists:
        raise ValueError('No SCAD available!')  # 如果OpenSCAD不可用,则抛出异常
    with MeshScript(meshes=meshes, script=script) as scad:
        result = scad.run(_scad_executable + ' $script -o $mesh_post')
    return result


def boolean(meshes, operation='difference'):
    '''
    对一组网格执行布尔操作

    :param meshes: 要进行布尔操作的网格列表
    :param operation: 布尔操作的类型,默认值为 'difference'
    :return: 执行结果,通常是处理后的网格数据
    '''
    script = operation + '(){'
    for i in range(len(meshes)):
        script += 'import(\"$mesh_' + str(i) + '\");'
    script += '}'
    return interface_scad(meshes, script)
