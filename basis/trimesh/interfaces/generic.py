import os
import platform
import subprocess
from string import Template
from tempfile import NamedTemporaryFile
from subprocess import check_call
from .. import io
from ..util import log


class MeshScript:
    """
    MeshScript类用于处理网格的临时文件创建和脚本执行

    :param meshes: 要处理的网格列表
    :param script: 要执行的脚本文本
    :param exchange: 网格文件的交换格式,默认值为'stl'
    :param debug: 如果为True,则启用调试模式,默认值为False
    :param kwargs: 传递给网格加载函数的其他参数
    """

    def __init__(self,
                 meshes,
                 script,
                 exchange='stl',
                 debug=False,
                 **kwargs):

        self.debug = debug
        self.kwargs = kwargs
        self.meshes = meshes
        self.script = script
        self.exchange = exchange

    def __enter__(self):
        """
        进入上下文管理器时调用,创建临时文件并准备脚本
        """
        # Windows在多个程序使用打开文件时可能会有问题,因此我们在enter调用结束时关闭它们,并在退出时自行删除
        # Blender按字母顺序对其对象进行排序,因此在文件名前加上网格编号前缀
        digit_count = len(str(len(self.meshes)))
        self.mesh_pre = [
            NamedTemporaryFile(
                suffix='.{}'.format(
                    self.exchange),
                prefix='{}_'.format(str(i).zfill(digit_count)),
                mode='wb',
                delete=False) for i in range(len(self.meshes))]
        self.mesh_post = NamedTemporaryFile(
            suffix='.{}'.format(
                self.exchange),
            mode='rb',
            delete=False)
        self.script_out = NamedTemporaryFile(
            mode='wb', delete=False)

        # 将网格导出到临时STL容器
        for mesh, file_obj in zip(self.meshes, self.mesh_pre):
            mesh.export(file_obj=file_obj.name)

        self.replacement = {'MESH_' + str(i): m.name
                            for i, m in enumerate(self.mesh_pre)}
        self.replacement['MESH_PRE'] = str(
            [i.name for i in self.mesh_pre])
        self.replacement['MESH_POST'] = self.mesh_post.name
        self.replacement['SCRIPT'] = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        if platform.system() == 'Windows':
            script_text = script_text.replace('\\', '\\\\')
        self.script_out.write(script_text.encode('utf-8'))

        # 关闭所有临时文件
        self.script_out.close()
        self.mesh_post.close()
        for file_obj in self.mesh_pre:
            file_obj.close()
        return self

    def run(self, command):
        """
        执行给定的命令并返回结果

        :param command: 要执行的命令
        :return: 从执行结果中加载的网格对象
        """
        command_run = Template(command).substitute(self.replacement).split()
        # 运行二进制文件 使用null避免资源警告
        with open(os.devnull, 'w') as devnull:
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            if self.debug:
                # 在调试模式下打印输出
                stdout = None
            else:
                stdout = devnull
            if self.debug:
                log.info('executing: {}'.format(' '.join(command_run)))
            check_call(command_run,
                       stdout=stdout,
                       stderr=subprocess.STDOUT,
                       startupinfo=startupinfo)
        # 将二进制结果作为Trimesh参数集返回
        mesh_results = io.load.load_mesh(self.mesh_post.name, **self.kwargs)
        return mesh_results

    def __exit__(self, *args, **kwargs):
        """
        退出上下文管理器时调用,删除临时文件
        """
        if self.debug:
            log.info('MeshScript.debug: not deleting {}'.format(self.script_out.name))
            return
        # 按名称删除所有临时文件
        # 它们已关闭,但其名称仍可用
        os.remove(self.script_out.name)
        for file_obj in self.mesh_pre:
            os.remove(file_obj.name)
        os.remove(self.mesh_post.name)
