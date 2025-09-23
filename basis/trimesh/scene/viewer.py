import pyglet
import pyglet.gl as gl
import numpy as np
from collections import deque
from ..transformations import Arcball

# 仅当面数量少于此时进行平滑
_SMOOTH_MAX_FACES = 100000


class SceneViewer(pyglet.window.Window):
    def __init__(self, scene, smooth=None, save_image=None, flags=None, resolution=(640, 480)):
        """
        初始化 SceneViewer 类的实例

        :param scene: 场景对象,包含要渲染的网格和灯光
        :param smooth: bool,是否平滑显示网格
        :param save_image: str,保存图像的路径,如果为 None 则不保存
        :param flags: dict,视图标志
        :param resolution: tuple,窗口分辨率,默认为 (640, 480)
        """
        self.scene = scene
        self.scene._redraw = self._redraw
        self.reset_view(flags=flags)
        visible = save_image is None
        width, height = resolution
        try:
            conf = gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
            super(SceneViewer, self).__init__(config=conf, visible=visible, resizable=True, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            conf = gl.Config(double_buffer=True)
            super(SceneViewer, self).__init__(config=conf, resizable=True, visible=visible, width=width, height=height)
        self.batch = pyglet.graphics.Batch()
        self._img = save_image
        self.vertex_list = {}
        self.vertex_list_md5 = {}
        for name, mesh in scene.meshes.items():
            self._add_mesh(name, mesh, smooth)
        self.init_gl()
        self.set_size(*resolution)
        self.update_flags()
        pyglet.app.run()

    def _redraw(self):
        # 重绘场景
        self.on_draw()

    def _update_meshes(self):
        #  更新场景中的网格
        for name, mesh in self.scene.meshes.items():
            md5 = mesh.md5() + mesh.visual.md5()
            if self.vertex_list_md5[name] != md5:
                self._add_mesh(name, mesh)

    def _add_mesh(self, name_mesh, mesh, smooth=None):
        """
        添加网格到场景中

        :param name_mesh: str,网格的名称
        :param mesh: Mesh,网格对象
        :param smooth: bool,是否平滑显示网格
        """
        if smooth is None:
            smooth = len(mesh.faces) < _SMOOTH_MAX_FACES
        if smooth:
            display = mesh.smoothed()
        else:
            display = mesh.copy()
            display.unmerge_vertices()
        self.vertex_list[name_mesh] = self.batch.add_indexed(*mesh_to_vertex_list(display))
        self.vertex_list_md5[name_mesh] = mesh.md5() + mesh.visual.md5()

    def reset_view(self, flags=None):
        """
        重置视图到基础状态

        :param flags: dict,视图标志
        """
        self.view = {'wireframe': False, 'cull': True, 'translation': np.zeros(3), 'center': self.scene.centroid,
                     'scale': self.scene.scale, 'ball': Arcball()}
        if isinstance(flags, dict):
            for k, v in flags.items():
                if k in self.view:
                    self.view[k] = v
        self.update_flags()

    def init_gl(self):
        """
        初始化 OpenGL 设置
        """
        gl.glClearColor(.93, .93, 1, 1)
        # glColor3f(1, 0, 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(.5, .5, 1, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(1, 0, .5, 0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, _gl_vector(0.192250, 0.192250, 0.192250))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, _gl_vector(0.507540, 0.507540, 0.507540))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, _gl_vector(.5082730, .5082730, .5082730))
        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, .4 * 128.0);
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def toggle_culling(self):
        """
        切换面剔除状态
        """
        self.view['cull'] = not self.view['cull']
        self.update_flags()

    def toggle_wireframe(self):
        """
        切换线框模式状态
        """
        self.view['wireframe'] = not self.view['wireframe']
        self.update_flags()

    def update_flags(self):
        """
        更新 OpenGL 渲染标志,根据当前视图设置调整渲染模式
        """
        if self.view['wireframe']:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        if self.view['cull']:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

    def on_resize(self, width, height):
        """
        处理窗口大小调整事件

        :param width: 新窗口宽度
        :param height: 新窗口高度
        """
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60., width / float(height), .01, 1000.)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        self.view['ball'].place([width / 2, height / 2], (width + height) / 2)

    def on_mouse_press(self, x, y, buttons, modifiers):
        """
        处理鼠标按下事件

        :param x: 鼠标 x 坐标
        :param y: 鼠标 y 坐标
        :param buttons: 按下的鼠标按钮
        :param modifiers: 修饰键
        """
        self.view['ball'].down([x, -y])

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        处理鼠标拖动事件

        :param x: 当前鼠标 x 坐标
        :param y: 当前鼠标 y 坐标
        :param dx: 鼠标 x 方向的拖动量
        :param dy: 鼠标 y 方向的拖动量
        :param buttons: 按下的鼠标按钮
        :param modifiers: 修饰键
        """
        delta = np.array([dx, dy], dtype=float) / [self.width, self.height]
        # 鼠标左键,control键向下(pan)
        if ((buttons == pyglet.window.mouse.LEFT) and
                (modifiers & pyglet.window.key.MOD_CTRL)):
            self.view['translation'][0:2] += delta
        # 鼠标左键,没有按修改键(旋转)
        elif (buttons == pyglet.window.mouse.LEFT):
            self.view['ball'].drag([x, -y])

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        处理鼠标滚动事件

        :param x: 鼠标 x 坐标
        :param y: 鼠标 y 坐标
        :param dx: 滚动 x 方向的量
        :param dy: 滚动 y 方向的量
        """
        self.view['translation'][2] += ((float(dy) / self.height) * self.view['scale'] * 5)

    def on_key_press(self, symbol, modifiers):
        """
        处理键盘按下事件

        :param symbol: 按下的键
        :param modifiers: 修饰键
        """
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()

    def on_draw(self):
        """
        渲染场景
        """
        self._update_meshes()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        # 从场景中获取新的相机变换
        transform_camera = self.scene.transforms['camera']
        # 将相机变换应用到矩阵堆栈
        gl.glMultMatrixf(_gl_matrix(transform_camera))
        # 拖动鼠标移动视图变换(但不改变场景)
        transform_view = _view_transform(self.view)
        gl.glMultMatrixf(_gl_matrix(transform_view))

        # 我们希望先渲染完全不透明的对象,然后是具有透明度的对象
        items = deque(self.scene.nodes.items())
        count_original = len(items)
        count = -1
        while len(items) > 0:
            count += 1
            item = items.popleft()
            name_node, name_mesh = item
            # 如果标志未定义,这将是 None 通过显式检查 False,使默认行为为渲染未定义标志的网格
            if self.node_flag(name_node, 'visible') == False:
                continue
            if self.scene.meshes[name_mesh].visual.transparency:
                # 将当前项放到队列的后面
                if count < count_original:
                    items.append(item)
                    continue
            transform = self.scene.transforms[name_node]
            # 将新矩阵添加到模型堆栈
            gl.glPushMatrix()
            # 通过节点变换进行变换
            gl.glMultMatrixf(_gl_matrix(transform))
            # 绘制应用了其变换的网格
            self.vertex_list[name_mesh].draw(mode=gl.GL_TRIANGLES)
            # 弹出矩阵堆栈,因为我们已经绘制了需要绘制的内容
            gl.glPopMatrix()

    def node_flag(self, node, flag):
        """
        获取节点的标志状态

        :param node: 节点名称
        :param flag: 标志名称
        :return: 标志状态
        """
        if flag in self.scene.flags[node]:
            return self.scene.flags[node][flag]
        return None

    def save_image(self, filename):
        """
        保存当前窗口的图像

        :param filename: 保存的文件名
        """
        colorbuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        colorbuffer.save(filename)

    def flip(self):
        """
        在事件循环中最后执行的函数,用于关闭窗口
        """
        super(self.__class__, self).flip()
        if self._img is not None:
            self.save_image(self._img)
            self.close()


def _view_transform(view):
    """
    根据视图参数计算变换矩阵

    :param view: 包含视图参数的字典
    :return: 变换矩阵
    """
    transform = view['ball'].matrix()
    transform[0:3, 3] = view['center']
    transform[0:3, 3] -= np.dot(transform[0:3, 0:3], view['center'])
    transform[0:3, 3] += view['translation'] * view['scale']
    return transform


def mesh_to_vertex_list(mesh, group=None):
    """
    将 Trimesh 对象转换为索引顶点列表构造函数的参数

    :param mesh: Trimesh 对象
    :param group: 顶点组
    :return: 顶点列表参数
    """
    mesh.visual.choose()
    normals = mesh.vertex_normals.reshape(-1).tolist()
    colors = mesh.visual.vertex_colors.reshape(-1).tolist()
    faces = mesh.faces.reshape(-1).tolist()
    vertices = mesh.vertices.reshape(-1).tolist()
    color_dimension = mesh.visual.vertex_colors.shape[1]
    color_type = 'c' + str(color_dimension) + 'B/static'
    args = (len(mesh.vertices),  # 顶点数量
            gl.GL_TRIANGLES,  # mode
            group,  # group
            faces,  # indices
            ('v3f/static', vertices),
            ('n3f/static', normals),
            (color_type, colors))
    return args


def _gl_matrix(array):
    """
    将 numpy 变换矩阵 (行主序, (4,4)) 转换为 GLfloat 变换矩阵 (列主序, (16,))

    :param array: numpy 变换矩阵
    :return: GLfloat 变换矩阵
    """
    a = np.array(array).T.reshape(-1)
    return (gl.GLfloat * len(a))(*a)


def _gl_vector(array, *args):
    """
    将数组和可选参数转换为 GLfloat 的平面向量

    :param array: 数组
    :param args: 可选参数
    :return: GLfloat 向量
    """
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector
