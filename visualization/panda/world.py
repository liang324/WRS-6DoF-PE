from unittest import loader
from direct.showbase.ShowBaseGlobal import globalClock, render2d
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import PerspectiveLens, OrthographicLens, AmbientLight, PointLight, Vec4, Vec3, Point3, \
    WindowProperties, Filename, NodePath, Shader, GraphicsPipe, FrameBufferProperties, GraphicsOutput
from direct.showbase.ShowBase import ShowBase
import visualization.panda.inputmanager as im
import visualization.panda.filter as flt
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletDebugNode
import os
import math
import time

from basis import data_adapter as p3dh
# from vision.pointcloud import o3dhelper as o3dh
import basis.robot_math as rm
import numpy as np
import visualization.panda.anime_info as ani


class World(ShowBase, object):
    def __init__(self,
                 cam_pos=np.array([2.0, 0.5, 2.0]),
                 lookat_pos=np.array([0, 0, 0.25]),
                 up=np.array([0, 0, 1]),
                 fov=40,
                 w=1920,
                 h=1080,
                 lens_type="perspective",
                 toggle_debug=False,
                 auto_cam_rotate=False):
        """
        World 类,继承自 ShowBase,用于设置和管理 Panda3D 的渲染环境

        :param cam_pos: 摄像机位置
        :param lookat_pos: 摄像机注视点位置
        :param up: 摄像机上方向
        :param fov: 摄像机视场角
        :param w: 窗口宽度
        :param h: 窗口高度
        :param lens_type: 镜头类型(透视或正交)
        :param toggle_debug: 是否开启调试模式
        :param auto_cam_rotate: 是否自动旋转摄像机

        author: weiwei
        date: 2015?, 20201115
        """
        # 初始化showbase父类后将taskMgr、loader、render2d等添加到builtin中
        super().__init__()
        # set up window
        winprops = WindowProperties(base.win.getProperties())
        winprops.setTitle("WRS机器人规划与控制系统!")
        base.win.requestProperties(winprops)
        self.disableAllAudio()  # 禁用所有音频
        self.setBackgroundColor(1, 1, 1)  # 设置背景颜色为白色 胡老师代码
        # self.setBackgroundColor(248 / 255, 244 / 255, 230 / 255) #  陈老师代码

        # 设置镜头
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(640, 480)  # 胡老师代码
            # lens.setFilmSize(1, 1)  # 陈老师代码

        # 禁用默认鼠标控制
        self.disableMouse()
        self.cam.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
        self.cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)
        # 设置光照 环境光
        ablight = AmbientLight("ambientlight")
        ablight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        self.ablightnode = self.cam.attachNewNode(ablight)
        self.render.setLight(self.ablightnode)

        # 点光源 1
        ptlight0 = PointLight("pointlight0")
        ptlight0.setColor(Vec4(1, 1, 1, 1))
        self._ptlightnode0 = self.cam.attachNewNode(ptlight0)
        self._ptlightnode0.setPos(0, 0, 0)
        self.render.setLight(self._ptlightnode0)

        # 点光源 2
        ptlight1 = PointLight("pointlight1")
        ptlight1.setColor(Vec4(.4, .4, .4, 1))
        self._ptlightnode1 = self.cam.attachNewNode(ptlight1)
        self._ptlightnode1.setPos(self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(self._ptlightnode1)
        # 点光源 3
        ptlight2 = PointLight("pointlight2")
        ptlight2.setColor(Vec4(.3, .3, .3, 1))
        self._ptlightnode2 = self.cam.attachNewNode(ptlight2)
        self._ptlightnode2.setPos(-self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(self._ptlightnode2)

        # 帮助工具
        self.p3dh = p3dh
        # self.o3dh = o3dh
        self.rbtmath = rm

        # 设置输入管理器
        self.lookatpos = lookat_pos
        self.inputmgr = im.InputManager(self, self.lookatpos)
        taskMgr.add(self._interaction_update, "interaction", appendTask=True)

        # 设置旋转摄像机
        if auto_cam_rotate:
            taskMgr.doMethodLater(.1, self._rotatecam_update, "rotate cam")

        # 设置窗口大小
        props = WindowProperties()
        if w is None or h is None:
            w = self.pipe.getDisplayWidth()
            h = self.pipe.getDisplayHeight()

        props.setSize(w, h)
        self.win.requestProperties(props)
        # outline edge shader
        # self.set_outlineshader()

        # 设置卡通效果
        self._separation = 1
        self.filter = flt.Filter(self.win, self.cam)
        self.filter.setCartoonInk(separation=self._separation)
        # self.filter.setViewGlow()

        # 设置物理世界
        self.physics_scale = 1e3
        self.physicsworld = BulletWorld()
        self.physicsworld.setGravity(Vec3(0, 0, -9.81 * self.physics_scale))
        taskMgr.add(self._physics_update, "physics", appendTask=True)
        globalbprrender = base.render.attachNewNode("globalbpcollider")
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        self._debugNP = globalbprrender.attachNewNode(debugNode)
        self._debugNP.show()
        self.toggledebug = toggle_debug
        if toggle_debug:
            self.physicsworld.setDebugNode(self._debugNP.node())
        self.physicsbodylist = []

        # 设置渲染更新(TODO,仅用于动态？)
        self._internal_update_obj_list = []  # 要绘制的节点路径、碰撞模型或子弹动力学模型
        self._internal_update_robot_list = []
        taskMgr.add(self._internal_update, "internal_update", appendTask=True)

        # 用于远程可视化
        self._external_update_objinfo_list = []  # see anime_info.py
        self._external_update_robotinfo_list = []
        taskMgr.add(self._external_update, "external_update", appendTask=True)

        # 用于静态模型
        self._noupdate_model_list = []

    def _interaction_update(self, task):
        """
        更新交互状态的方法

        :param task: 当前任务对象.
        :return: 返回 task.cont 以继续任务循环.
        """
        # 重置宽高比
        aspectRatio = self.getAspectRatio()
        self.cam.node().getLens().setAspectRatio(aspectRatio)

        # 检查鼠标交互事件
        self.inputmgr.check_mouse1drag()
        self.inputmgr.check_mouse2drag()
        self.inputmgr.check_mouse3click()
        self.inputmgr.check_mousewheel()
        self.inputmgr.check_resetcamera()
        return task.cont

    def _physics_update(self, task):
        """
        在物理世界中执行物理模拟

        :param task: 当前任务对象
        :return: 返回 task.cont 以继续任务循环
        """
        # 执行物理模拟
        self.physicsworld.doPhysics(globalClock.getDt(), 20, 1 / 1200)
        return task.cont

    def _internal_update(self, task):
        """
        更新内部机器人和对象的状态

        :param task: 当前任务对象
        :return: 返回 task.cont 以继续任务循环
        """
        # 更新机器人状态
        for robot in self._internal_update_robot_list:
            robot.detach()  # TODO 生成网格模型？
            robot.attach_to(self)
        for obj in self._internal_update_obj_list:
            obj.detach()
            obj.attach_to(self)
        return task.cont

    def _rotatecam_update(self, task):
        """
        根据相机位置和观察点位置计算相机的旋转角度,并更新相机的位置和朝向

        :param task: 当前任务对象
        :return: 返回 task.cont 以继续任务循环
        """
        # 计算相机角度
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1] - self.lookatpos[1], campos[0] - self.lookatpos[0])
        # 调整相机角度
        if camangle < 0:
            camangle += math.pi * 2
        if camangle >= math.pi * 2:
            camangle = 0
        else:
            camangle += math.pi / 360
        # 计算相机半径
        camradius = math.sqrt((campos[0] - self.lookatpos[0]) ** 2 + (campos[1] - self.lookatpos[1]) ** 2)

        # 更新相机位置
        camx = camradius * math.cos(camangle)
        camy = camradius * math.sin(camangle)
        self.cam.setPos(self.lookatpos[0] + camx, self.lookatpos[1] + camy, campos[2])
        # 更新相机朝向
        self.cam.lookAt(self.lookatpos[0], self.lookatpos[1], self.lookatpos[2])
        return task.cont

    def _external_update(self, task):
        """
        更新外部对象和机器人的状态

        :param task: 任务对象
        :return: 继续任务
        """
        for _external_update_robotinfo in self._external_update_robotinfo_list:
            # 更新机器人信息
            robot_s = _external_update_robotinfo.robot_s
            robot_component_name = _external_update_robotinfo.robot_component_name
            robot_meshmodel = _external_update_robotinfo.robot_meshmodel
            robot_meshmodel_parameter = _external_update_robotinfo.robot_meshmodel_parameters
            robot_path = _external_update_robotinfo.robot_path
            robot_path_counter = _external_update_robotinfo.robot_path_counter

            # 分离当前的机器人模型
            robot_meshmodel.detach()

            # 更新机器人关节状态
            robot_s.fk(component_name=robot_component_name, jnt_values=robot_path[robot_path_counter])

            # 生成新的机器人模型
            _external_update_robotinfo.robot_meshmodel = robot_s.gen_meshmodel(
                tcp_jntid=robot_meshmodel_parameter[0],
                tcp_loc_pos=robot_meshmodel_parameter[1],
                tcp_loc_rotmat=robot_meshmodel_parameter[2],
                toggle_tcpcs=robot_meshmodel_parameter[3],
                toggle_jntscs=robot_meshmodel_parameter[4],
                rgba=robot_meshmodel_parameter[5],
                name=robot_meshmodel_parameter[6])

            # 附加新的机器人模型
            _external_update_robotinfo.robot_meshmodel.attach_to(self)

            # 更新路径计数器
            _external_update_robotinfo.robot_path_counter += 1
            if _external_update_robotinfo.robot_path_counter >= len(robot_path):
                _external_update_robotinfo.robot_path_counter = 0

        for _external_update_objinfo in self._external_update_objinfo_list:
            # 更新对象信息
            obj = _external_update_objinfo.obj
            obj_parameters = _external_update_objinfo.obj_parameters
            obj_path = _external_update_objinfo.obj_path
            obj_path_counter = _external_update_objinfo.obj_path_counter

            # 分离当前的对象
            obj.detach()
            # 更新对象位置和旋转矩阵
            obj.set_pos(obj_path[obj_path_counter][0])
            obj.set_rotmat(obj_path[obj_path_counter][1])
            # 设置对象颜色
            obj.set_rgba(obj_parameters[0])
            # 附加对象
            obj.attach_to(self)
            # 更新路径计数器
            _external_update_objinfo.obj_path_counter += 1
            if _external_update_objinfo.obj_path_counter >= len(obj_path):
                _external_update_objinfo.obj_path_counter = 0
        return task.cont

    def change_debugstatus(self, toggledebug):
        """
        改变调试状态

        :param toggledebug: 是否启用调试
        :return: None
        """
        if self.toggledebug == toggledebug:
            return
        elif toggledebug:
            self.physicsworld.setDebugNode(self._debugNP.node())
        else:
            self.physicsworld.clearDebugNode()
        self.toggledebug = toggledebug

    def attach_internal_update_obj(self, obj):
        """
        附加内部更新对象

        :param obj: CollisionModel 或 (Static)GeometricModel
        :return: None
        """
        self._internal_update_obj_list.append(obj)

    def detach_internal_update_obj(self, obj):
        """
        分离内部更新对象

        :param obj: 目标对象
        :return: None
        """
        self._internal_update_obj_list.remove(obj)
        obj.detach()

    def clear_internal_update_obj(self):
        """
        清除所有内部更新对象

        :return: None
        """
        tmp_internal_update_obj_list = self._internal_update_obj_list.copy()
        self._internal_update_obj_list = []
        for obj in tmp_internal_update_obj_list:
            obj.detach()

    def attach_internal_update_robot(self, robot_meshmodel):
        """
        附加内部更新机器人

        :param robot_meshmodel: 机器人模型
        :return: None
        """
        self._internal_update_robot_list.append(robot_meshmodel)

    def detach_internal_update_robot(self, robot_meshmodel):
        """
        分离内部更新机器人

        :param robot_meshmodel: 机器人模型
        :return: None
        """
        tmp_internal_update_robot_list = self._internal_update_robot_list.copy()
        self._internal_update_robot_list = []
        for robot in tmp_internal_update_robot_list:
            robot.detach()

    def clear_internal_update_robot(self):
        """
        清除所有内部更新机器人

        :return: None
        """
        for robot in self._internal_update_robot_list:
            self.detach_internal_update_robot(robot)

    def attach_external_update_obj(self, objinfo):
        """
        附加外部更新对象信息

        :param objinfo: anime_info.ObjInfo 对象
        :return: None
        """
        self._external_update_objinfo_list.append(objinfo)

    def detach_external_update_obj(self, obj_info):
        """
        分离外部更新对象

        :param obj_info: anime_info.ObjInfo 对象
        :return: None
        """
        self._external_update_objinfo_list.remove(obj_info)
        obj_info.obj.detach()

    def clear_external_update_obj(self):
        """
        清除所有外部更新对象

        :return: None
        """
        for obj in self._external_update_objinfo_list:
            self.detach_external_update_obj(obj)

    def attach_external_update_robot(self, robotinfo):
        """
        附加外部更新机器人信息

        :param robotinfo: anime_info.RobotInfo 对象
        :return: None
        """
        self._external_update_robotinfo_list.append(robotinfo)

    def detach_external_update_robot(self, robot_info):
        """
        分离外部更新机器人

        :param robot_info: anime_info.RobotInfo 对象
        :return: None
        """
        self._external_update_robotinfo_list.remove(robot_info)
        robot_info.robot_meshmodel.detach()

    def clear_external_update_robot(self):
        """
        清除所有外部更新机器人

        :return: None
        """
        for robot in self._external_update_robotinfo_list:
            self.detach_external_update_robot(robot)

    def attach_noupdate_model(self, model):
        """
        附加不更新的模型

        :param model: 模型对象
        :return: None
        """
        model.attach_to(self)
        self._noupdate_model_list.append(model)

    def detach_noupdate_model(self, model):
        """
        分离不更新的模型

        :param model: 模型对象
        :return: None
        """
        model.detach()
        self._noupdate_model_list.remove(model)

    def clear_noupdate_model(self):
        """
        清除所有不更新的模型

        :return: None
        """
        for model in self._noupdate_model_list:
            model.detach()
        self._noupdate_model_list = []

    def change_campos(self, campos):
        """
        改变摄像机位置

        :param campos: 摄像机位置,格式为 [x, y, z]
        :return: None
        """
        self.cam.setPos(campos[0], campos[1], campos[2])
        self.inputmgr = im.InputManager(self, self.lookatpos)

    def change_lookatpos(self, lookatpos):
        """
        改变摄像机的观察点位置

        :param lookatpos: 观察点位置,格式为 [x, y, z]
        :return: None

        author: weiwei
        date: 20180606
        """
        self.cam.lookAt(lookatpos[0], lookatpos[1], lookatpos[2])
        self.lookatpos = lookatpos
        self.inputmgr = im.InputManager(self, self.lookatpos)

    def change_campos_and_lookat_pos(self, cam_pos, lookat_pos):
        """
        同时改变摄像机位置和观察点位置

        :param cam_pos: 摄像机位置,格式为 [x, y, z]
        :param lookat_pos: 观察点位置,格式为 [x, y, z]
        :return: None
        """
        self.cam.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
        self.cam.lookAt(lookat_pos[0], lookat_pos[1], lookat_pos[2])
        self.lookatpos = lookat_pos
        self.inputmgr = im.InputManager(self, self.lookatpos)

    def set_cartoonshader(self, switchtoon=False):
        """
        设置卡通着色器

        :param switchtoon: 是否启用卡通着色器
        :return: None

        author: weiwei
        date: 20180601
        """
        this_dir, this_filename = os.path.split(__file__)
        if switchtoon:
            lightinggen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "lighting_gen.sha"))
            tempnode = NodePath("temp")
            tempnode.setShader(loader.loadShader(lightinggen))
            self.cam.node().setInitialState(tempnode.getState())
            # self.render.setShaderInput("light", self.cam)
            self.render.setShaderInput("light", self._ablightnode)

        normalsBuffer = self.win.makeTextureBuffer("normalsBuffer", 0, 0)
        normalsBuffer.setClearColor(Vec4(0.5, 0.5, 0.5, 1))
        normalsCamera = self.makeCamera(normalsBuffer, lens=self.cam.node().getLens(), scene=self.render)
        normalsCamera.reparentTo(self.cam)
        normalgen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "normal_gen.sha"))
        tempnode = NodePath("temp")
        tempnode.setShader(loader.loadShader(normalgen))
        normalsCamera.node().setInitialState(tempnode.getState())
        drawnScene = normalsBuffer.getTextureCard()
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.reparentTo(render2d)
        self.drawnScene = drawnScene
        self.separation = 0.001
        self.cutoff = 0.05
        inkGen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "ink_gen.sha"))
        drawnScene.setShader(loader.loadShader(inkGen))
        drawnScene.setShaderInput("separation", Vec4(0, 0, self.separation, 0))
        drawnScene.setShaderInput("cutoff", Vec4(self.cutoff))

    def set_outlineshader(self):
        """
        设置轮廓着色器

        文档 1: https://qiita.com/nmxi/items/bfd10a3b3f519878e74e
        文档 2: https://docs.panda3d.org/1.10/python/programming/shaders/list-of-cg-inputs
        :return: None

        author: weiwei
        date: 20180601, 20201210osaka
        """
        depth_sha = """
        void vshader(float4 vtx_position : POSITION,
                     float4 vtx_normal : NORMAL,
                     uniform float4x4 mat_modelproj,
                     uniform float4x4 mat_modelview,
                     out float4 l_position : POSITION,
                     out float4 l_color0: COLOR0) {
            l_position = mul(mat_modelproj, vtx_position);
            float depth = l_position.a*.1;
            //l_color0 = vtx_position + float4(depth, depth, depth, 1);
            l_color0 = float4(depth, depth, depth, 1);
        }
        void fshader(float4 l_color0: COLOR0,
                     uniform sampler2D tex_0 : TEXUNIT0,
                     out float4 o_color : COLOR) {
            o_color = l_color0;
        }"""
        outline_sha = """
        void vshader(float4 vtx_position : POSITION,
             float2 vtx_texcoord0 : TEXCOORD0,
             uniform float4x4 mat_modelproj,
             out float4 l_position : POSITION,
             out float2 l_texcoord0 : TEXCOORD0)
        {
          l_position = mul(mat_modelproj, vtx_position);
          l_texcoord0 = vtx_texcoord0;
        }
        void fshader(float2 l_texcoord0 : TEXCOORD0,
                     uniform sampler2D tex_0 : TEXUNIT0,
                     uniform float2 sys_windowsize,
                     out float4 o_color : COLOR)
        {
          float sepx = 1/sys_windowsize.x;
          float sepy = 1/sys_windowsize.y;
          float4 color0 = tex2D(tex_0, l_texcoord0);
          float2 texcoord1 = l_texcoord0+float2(sepx, 0);
          float4 color1 = tex2D(tex_0, texcoord1);
          float2 texcoord2 = l_texcoord0+float2(0, sepy);
          float4 color2 = tex2D(tex_0, texcoord2);
          float2 texcoord3 = l_texcoord0+float2(-sepx, 0);
          float4 color3 = tex2D(tex_0, texcoord3);
          float2 texcoord4 = l_texcoord0+float2(0, -sepy);
          float4 color4 = tex2D(tex_0, texcoord4);
          float2 texcoord5 = l_texcoord0+float2(sepx, sepy);
          float4 color5 = tex2D(tex_0, texcoord5);
          float2 texcoord6 = l_texcoord0+float2(-sepx, -sepy);
          float4 color6 = tex2D(tex_0, texcoord6);
          float2 texcoord7 = l_texcoord0+float2(-sepx, sepy);
          float4 color7 = tex2D(tex_0, texcoord7);
          float2 texcoord8 = l_texcoord0+float2(sepx, -sepy);
          float4 color8 = tex2D(tex_0, texcoord8);
          float2 texcoord9 = l_texcoord0+float2(2*sepx, 0);
          float4 color9 = tex2D(tex_0, texcoord9);
          float2 texcoord10 = l_texcoord0+float2(-2*sepx, 0);
          float4 color10 = tex2D(tex_0, texcoord10);
          float2 texcoord11 = l_texcoord0+float2(0, 2*sepy);
          float4 color11 = tex2D(tex_0, texcoord11);
          float2 texcoord12 = l_texcoord0+float2(0, -2*sepy);
          float4 color12 = tex2D(tex_0, texcoord12);
          o_color = (color0-color1).x > .005 || (color0-color2).x > .005 || (color0-color3).x > .005 ||
                    (color0-color4).x > .005 || (color0-color5).x > .005 || (color0-color6).x > .005 ||
                    (color0-color7).x > .005 || (color0-color8).x > .005 || (color0-color9).x > .005 ||
                    (color0-color10).x > .005 || (color0-color11).x > .005 || (color0-color12).x > .005 ?
                    float4(0, 0, 0, 1) : float4(0, 0, 0, 0);
        }"""
        depthBuffer = self.win.makeTextureBuffer("depthBuffer", 0, 0)
        depthBuffer.setClearColor(Vec4(1, 1, 1, 1))
        depthCamera = self.makeCamera(depthBuffer, lens=self.cam.node().getLens(), scene=self.render)
        depthCamera.reparentTo(self.cam)
        tempnode = NodePath("depth")
        tempnode.setShader(Shader.make(depth_sha, Shader.SL_Cg))
        depthCamera.node().setInitialState(tempnode.getState())
        drawnScene = depthBuffer.getTextureCard()
        drawnScene.reparentTo(render2d)
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.setShader(Shader.make(outline_sha, Shader.SL_Cg))
