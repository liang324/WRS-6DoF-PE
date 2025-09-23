from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename, Texture
from panda3d.core import AmbientLight, DirectionalLight, PointLight
from panda3d.core import NodePath, TextNode
from direct.task.Task import Task
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
import sys
import os
import random


def addInstructions(pos, msg):
    """
    在屏幕上显示指令

    :param pos: 指令在屏幕上的垂直位置(相对于左上角)
    :param msg: 要显示的指令文本
    :return: 返回一个 OnscreenText 对象,用于在屏幕上显示文本
    """
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=base.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)


def addTitle(text):
    """
    在屏幕上显示标题

    :param text: 要显示的标题文本
    :return: 返回一个 OnscreenText 对象,用于在屏幕上显示文本
    """
    return OnscreenText(text=text, style=1, fg=(1, 1, 1, 1), scale=.08,
                        parent=base.a2dBottomRight, align=TextNode.ARight,
                        pos=(-0.1, 0.09), shadow=(0, 0, 0, 1))


random.seed()

class TeapotOnTVDemo(ShowBase):
    """
    TeapotOnTVDemo 类用于演示如何使用 Panda3D 的渲染到纹理功能
    该类创建一个场景,其中包含一个旋转的茶壶,并在多个 TV 人物的屏幕上显示该茶壶的渲染结果
    """

    def __init__(self):
        """
        初始化 TeapotOnTVDemo 类的实例

        该构造函数设置场景的背景颜色,添加标题和指令,并配置渲染到纹理的缓冲区和场景图
        """
        ShowBase.__init__(self)
        self.disableMouse()
        self.setBackgroundColor((0, 0, 0, 1))

        # 发布指令
        self.title = addTitle("Panda3D: Tutorial - Using Render-to-Texture")
        self.inst1 = addInstructions(0.06, "ESC: Quit")  # ESC: 退出
        self.inst2 = addInstructions(0.12, "Up/Down: Zoom in/out on the Teapot")  # 上/下: 放大/缩小茶壶
        self.inst3 = addInstructions(0.18, "Left/Right: Move teapot left/right")  # 左/右: 左右移动茶壶
        self.inst4 = addInstructions(0.24, "V: View the render-to-texture results")  # V: 查看渲染到纹理的结果

        # 创建一个用于保存新场景纹理的缓冲区
        altBuffer = self.win.makeTextureBuffer("hello", 256, 256)

        # 设置新的场景图
        altRender = NodePath("new render")

        # 设置相机
        self.altCam = self.makeCamera(altBuffer)
        self.altCam.reparentTo(altRender)
        self.altCam.setPos(0, -10, 0)

        # 加载茶壶模型并设置动画
        self.teapot = loader.loadModel('teapot')
        self.teapot.reparentTo(altRender)
        self.teapot.setPos(0, 0, -1)
        self.teapot.hprInterval(1.5, (360, 360, 360)).loop()  # 旋转动画

        # 为茶壶添加光照
        dlight = DirectionalLight('dlight')  # 定向光
        alight = AmbientLight('alight')  # 环境光
        dlnp = altRender.attachNewNode(dlight)
        alnp = altRender.attachNewNode(alight)
        dlight.setColor((0.8, 0.8, 0.5, 1))
        alight.setColor((0.2, 0.2, 0.2, 1))
        dlnp.setHpr(0, -60, 0)
        altRender.setLight(dlnp)
        altRender.setLight(alnp)

        # 为主场景添加光照
        plight = PointLight('plight')  # 点光源
        plnp = render.attachNewNode(plight)
        plnp.setPos(0, 0, 10)
        render.setLight(plnp)
        render.setLight(alnp)

        # Panda包含一个内置的查看器,可以让你查看结果
        # 你的渲染到纹理操作.这段代码配置查看器

        # 配置内置的缓冲区查看器
        self.accept("v", self.bufferViewer.toggleEnable)
        self.accept("V", self.bufferViewer.toggleEnable)
        self.bufferViewer.setPosition("llcorner")
        self.bufferViewer.setCardSize(1.0, 0.0)

        # 创建 TV 人物,每个 TV 人物将显示离屏纹理
        self.tvMen = []
        self.makeTvMan(-5, 30, 1, altBuffer.getTexture(), 0.9)
        self.makeTvMan(5, 30, 1, altBuffer.getTexture(), 1.4)
        self.makeTvMan(0, 23, -3, altBuffer.getTexture(), 2.0)
        self.makeTvMan(-5, 20, -6, altBuffer.getTexture(), 1.1)
        self.makeTvMan(5, 18, -5, altBuffer.getTexture(), 1.7)

        # 接受用户输入
        self.accept("escape", sys.exit, [0])  # ESC: 退出
        self.accept("arrow_up", self.zoomIn)  # 上箭头: 放大
        self.accept("arrow_down", self.zoomOut)  # 下箭头: 缩小
        self.accept("arrow_left", self.moveLeft)  # 左箭头: 向左移动
        self.accept("arrow_right", self.moveRight)  # 右箭头: 向右移动

    def makeTvMan(self, x, y, z, tex, playrate):
        """
        创建一个 TV 人物并将其放置在指定位置

        :param x: TV 人物的 x 坐标
        :param y: TV 人物的 y 坐标
        :param z: TV 人物的 z 坐标
        :param tex: 要在 TV 屏幕上显示的纹理
        :param playrate: TV 人物动画的播放速率
        :return: None
        """
        man = Actor()
        man.loadModel('models/mechman_idle')
        man.setPos(x, y, z)
        man.reparentTo(render)
        faceplate = man.find("**/faceplate")
        faceplate.setTexture(tex, 1)
        man.setPlayRate(playrate, "mechman_anim")
        man.loop("mechman_anim")
        self.tvMen.append(man)

    def zoomIn(self):
        """
        缩小视角,使相机向茶壶靠近
        """
        self.altCam.setY(self.altCam.getY() * 0.9)

    def zoomOut(self):
        """
        放大视角,使相机远离茶壶
        """
        self.altCam.setY(self.altCam.getY() * 1.2)

    def moveLeft(self):
        """
        将相机向左移动
        """
        self.altCam.setX(self.altCam.getX() + 1)

    def moveRight(self):
        """
        将相机向右移动
        """
        self.altCam.setX(self.altCam.getX() - 1)


demo = TeapotOnTVDemo()
demo.run()
