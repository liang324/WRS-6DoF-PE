import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import Texture, PerspectiveLens
from typing import List
import matplotlib.pyplot as plt
from panda3d.core import ConfigVariableString
from sensors.cameras import CameraIntrinsics

ConfigVariableString('background-color', '1.0 1.0 1.0 0.0')  # 设置背景为白色


class SceneSimulator(ShowBase):
    """
    模拟一个包含图像和深度相机的场景
    """
    IMAGE_CAMERA_RENDER_ORDER = -2
    DEPTH_CAMERA_RENDER_ORDER = -2

    def __init__(self):
        """
        模拟一个没有相机的空场景

        注意:可以通过后续调用 `add_image_camera` 和 `add_depth_camera` 来添加相机
        """
        ShowBase.__init__(self)
        self._image_buffers = []  # 调用父类构造函数
        self._image_cameras = []  # 存储图像缓冲区的列表
        self._depth_buffers = []  # 存储图像相机的列表
        self._depth_cameras = []  # 存储深度相机的列表

    def render_frame(self):
        """
        渲染当前帧
        """
        self.graphics_engine.render_frame()  # 调用图形引擎渲染帧

    def add_image_camera(self, intrinsics: CameraIntrinsics, pos, hpr, name=None):
        """
        添加图像相机

        :param intrinsics: 相机的内参,包含相机的视场、分辨率等信息
        :param pos: 相机的位置(x, y, z)
        :param hpr: 相机的朝向(航向、俯仰、滚转)
        :param name: 相机的名称(可选)
        """
        # 设置纹理和图形缓冲区
        window_props = WindowProperties.size(*intrinsics.get_size())  # 设置窗口属性
        frame_buffer_props = FrameBufferProperties()  # 创建帧缓冲属性
        buffer = self.graphicsEngine.make_output(self.pipe,
                                                 f'Image Buffer [{name}]',  # 创建图像缓冲区
                                                 self.IMAGE_CAMERA_RENDER_ORDER,
                                                 frame_buffer_props,
                                                 window_props,
                                                 GraphicsPipe.BFRefuseWindow,  # 不打开窗口
                                                 self.win.getGsg(),
                                                 self.win
                                                 )
        texture = Texture()  # 创建纹理
        buffer.addRenderTexture(texture, GraphicsOutput.RTMCopyRam)  # 将纹理添加到缓冲区

        # 根据相机内参设置镜头
        lens = PerspectiveLens()  # 创建透视镜头
        lens.set_film_size(*intrinsics.get_size())  # 设置胶卷大小
        lens.set_fov(*np.rad2deg(intrinsics.get_fov()))  # 设置视场

        camera = self.makeCamera(buffer, lens=lens, camName=f'Image Camera [{name}]')  # 创建相机
        camera.reparentTo(self.render)  # 将相机附加到渲染节点
        camera.setPos(*pos)  # 设置相机位置
        camera.setHpr(*hpr)  # 设置相机朝向

        self._image_buffers.append(buffer)  # 将缓冲区添加到列表
        self._image_cameras.append(camera)  # 将相机添加到列表

    def get_images(self) -> List[np.ndarray]:
        """
        获取每个图像相机在最近一次渲染后的图像

        注意:必须单独调用 `self.render_frame()

        :return: 返回一个包含所有图像的列表,每个图像为一个 NumPy 数组
        """
        images = []  # 存储图像的列表
        for buffer in self._image_buffers:  # 遍历所有图像缓冲区
            tex = buffer.getTexture()  # 获取纹理

            data = tex.getRamImage()  # 获取纹理的 RAM 图像数据
            image = np.frombuffer(data, np.uint8)  # 将数据转换为 NumPy 数组
            image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())  # 设置图像形状
            image = np.flipud(image)  # 垂直翻转图像
            images.append(image)  # 将图像添加到列表
        return images


if __name__ == '__main__':
    sim = SceneSimulator()

    n_cameras = 3
    camera_radius = 6
    camera_height = 4
    intrinsics = CameraIntrinsics.from_size_and_fov((1920, 1080), (np.pi / 6, np.pi / 6))
    for i in range(n_cameras):
        camera_angle = i * (2 * np.pi / n_cameras)
        pos = (camera_radius * np.sin(camera_angle), - camera_radius * np.cos(camera_angle), camera_height)
        sim.add_image_camera(intrinsics, pos, (0, 0, 0), name=str(i))
        sim._image_cameras[i].lookAt(sim.render)

    # place box in scene
    x, y, r = 0, 0, 1
    box = sim.loader.loadModel("models/box")
    box.reparentTo(sim.render)
    box.setScale(r)
    box.setPos(x - r / 2, y - r / 2, 0)
    sim.box = box

    sim.render_frame()
    observations = sim.get_images()

    for obs in observations:
        plt.imshow(obs)
        plt.show()
