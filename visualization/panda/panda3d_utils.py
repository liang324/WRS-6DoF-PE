# Panda3d扩展
# Author: Hao Chen


import functools

import numpy as np
import cv2
from panda3d.core import (Texture,
                          NodePath,
                          WindowProperties,
                          Vec3,
                          Point3,
                          PerspectiveLens,
                          OrthographicLens,
                          PGTop,
                          PGMouseWatcherBackground)
from direct.gui.OnscreenImage import OnscreenImage

import visualization.panda.filter as flt
import visualization.panda.inputmanager as im
import visualization.panda.world as wd


def img_to_n_channel(img, channel=3):
    """
    重复一个通道n次并堆叠
    :param img: 输入图像
    :param channel: 通道数
    :return: 返回堆叠后的图像
    """
    return np.stack((img,) * channel, axis=-1)


def letter_box(img, new_shape=(640, 640), color=(.45, .45, .45), auto=True, scale_fill=False, scale_up=True, stride=32):
    """
    该函数来自YOLOv5 (https://github.com/ultralytics/yolov5)
    """
    # 调整图像大小并填充以满足步幅倍数约束
    shape = img.shape[:2]  # 当前形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # 仅缩小,不放大(以获得更好的验证mAP)
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度,高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 宽高填充
    elif scale_fill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度,高度比例

    dw /= 2  # 将填充分为两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(color[0] * 255, color[0] * 255, color[0] * 255))  # 添加边框
    return img


class ImgOnscreen(object):
    """
    在Showbase的2D场景中添加一个屏幕图像
    """

    def __init__(self, size, parent_np=None):
        """
         :param size: (宽度, 高度)
        :param parent_np: 应该是ShowBase或ExtraWindow
        """
        self._size = size
        self.tx = Texture("video")
        self.tx.setup2dTexture(size[0], size[1], Texture.TUnsignedByte, Texture.FRgb8)
        # 这会进行一些重要的设置调用
        # self.tx.load(PNMImage(card_size[0], card_size[1]))
        self.onscreen_image = OnscreenImage(self.tx,
                                            pos=(0, 0, 0),
                                            parent=parent_np.render2d)

    def update_img(self, img: np.ndarray):
        """
        更新屏幕上的图像
        :param img: 输入图像
        :return: 无返回值
        """
        if img.shape[2] == 1:
            img = img_to_n_channel(img)
        resized_img = letter_box(img, new_shape=[self._size[1], self._size[0]], auto=False)
        self.tx.setRamImage(resized_img.tostring())

    def remove(self):
        """
        释放内存
        :return: 无返回值
        """
        if self.onscreen_image is not None:
            self.onscreen_image.destroy()

    def __del__(self):
        self.remove()


class ExtraWindow(object):
    """
    在场景中创建一个额外的窗口
    TODO: 需要修复的小错误:win.requestProperties不能立即更改窗口属性
    :return: 无返回值
    """

    def __init__(self, base: wd.World,
                 window_title: str = "WRS机器人规划和控制系统",
                 cam_pos: np.ndarray = np.array([2.0, 0, 2.0]),
                 lookat_pos: np.ndarray = np.array([0, 0, 0.25]),
                 up: np.ndarray = np.array([0, 0, 1]),
                 fov: int = 40,
                 w: int = 1920,
                 h: int = 1080,
                 lens_type: str = "perspective"):
        self._base = base
        # 设置渲染场景
        self.render = NodePath("extra_win_render")
        # 设置渲染2D
        self.render2d = NodePath("extra_win_render2d")
        self.render2d.setDepthTest(0)
        self.render2d.setDepthWrite(0)

        self.win = base.openWindow(props=WindowProperties(base.win.getProperties()),
                                   makeCamera=False,
                                   scene=self.render,
                                   requireWindow=True, )

        # 将窗口背景设置为白色
        base.setBackgroundColor(r=1, g=1, b=1, win=self.win)
        # 设置窗口标题和窗口尺寸
        self.set_win_props(title=window_title, size=(w, h), )
        # 设置镜头并为新窗口设置相机
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1, 1)

        # 使宽高比与基础窗口相同
        aspect_ratio = base.getAspectRatio()
        lens.setAspectRatio(aspect_ratio)
        self.cam = base.makeCamera(self.win, scene=self.render, )  # 也可以在base.camList中找到
        self.cam.reparentTo(self.render)
        self.cam.setPos(Point3(cam_pos[0], cam_pos[1], cam_pos[2]))
        self.cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)  # 使用与系统相同的镜头

        # 设置卡通效果
        self._separation = 1
        self.filter = flt.Filter(self.win, self.cam)
        self.filter.setCartoonInk(separation=self._separation)

        # 相机在相机2D中
        self.cam2d = base.makeCamera2d(self.win, )
        self.cam2d.reparentTo(self.render2d)
        # 将GPTop附加到render2d以确保可以使用DirectGui
        self.aspect2d = self.render2d.attachNewNode(PGTop("aspect2d"))
        # self.aspect2d.setScale(1.0 / aspect_ratio, 1.0, 1.0)

        #  为新窗口设置鼠标
        #  鼠标观察者的名称是为了适应输入管理器中的名称
        self.mouse_thrower = base.setupMouse(self.win, fMultiWin=True)
        self.mouseWatcher = self.mouse_thrower.getParent()
        self.mouseWatcherNode = self.mouseWatcher.node()
        self.aspect2d.node().setMouseWatcher(self.mouseWatcherNode)
        # self.mouseWatcherNode.addRegion(PGMouseWatcherBackground())

        # 设置输入管理器
        self.inputmgr = im.InputManager(self, lookatpos=lookat_pos)

        # 从基础复制属性和函数
        # 将绑定函数更改为函数,并绑定到`self`以成为未绑定函数
        self._interaction_update = functools.partial(base._interaction_update.__func__, self)
        self.p3dh = base.p3dh

        base.taskMgr.add(self._interaction_update, "interaction_extra_window", appendTask=True)

    @property
    def size(self):
        """
        获取窗口的尺寸

        :return: 窗口尺寸,类型为numpy数组
        """
        size = self.win.getProperties().size
        return np.array([size[0], size[1]])

    def getAspectRatio(self):
        """
        获取窗口的宽高比

        :return: 窗口的宽高比
        """
        return self._base.getAspectRatio(self.win)

    def set_win_props(self,
                      title: str,
                      size: tuple):
        """
        设置额外窗口的属性
        :param title: 窗口标题
        :param size: 1x2元组描述宽度和高度
        :return: 无返回值
        """
        win_props = WindowProperties()
        win_props.setSize(size[0], size[1])
        win_props.setTitle(title)
        self.win.requestProperties(win_props)

    def set_origin(self, origin: np.ndarray):
        """
        设置窗口的原点位置

        :param origin: 1x2 numpy数组描述窗口的左上角
        :return: 无返回值
        """
        win_props = WindowProperties()
        win_props.setOrigin(origin[0], origin[1])
        self.win.requestProperties(win_props)


if __name__ == "__main__":
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame(length=.2).attach_to(base)

    # extra window 1
    ew = ExtraWindow(base, cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    ew.set_origin((0, 40))
    # ImgOnscreen()
    img = cv2.imread("img.png")
    on_screen_img = ImgOnscreen(img.shape[:2][::-1], parent_np=ew)
    on_screen_img.update_img(img)

    # extra window 2
    ew2 = ExtraWindow(base, cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    ew2.set_origin((0, ew.size[1]))
    gm.gen_frame(length=.2).objpdnp.reparentTo(ew2.render)

    base.run()
