from direct.filter import FilterManager as pfm
from panda3d.core import Texture, CardMaker, NodePath, AuxBitplaneAttrib, LightRampAttrib, Camera, OrthographicLens, \
    GraphicsOutput, WindowProperties, FrameBufferProperties, GraphicsPipe


class FilterManager(pfm.FilterManager):
    """
    FilterManager 类用于管理和应用图像滤波和后处理效果,特别是在 Panda3D 引擎中

    继承自 direct.filter.FilterManager,并提供额外的功能来配置渲染缓冲区和应用滤波效果
    """

    def __init__(self, win, cam):
        super().__init__(win, cam)

    def renderSceneInto(self, depthtex=None, colortex=None, auxtex=None, auxbits=0, textures=None, fbprops=None,
                        clamping=None):
        """
        将场景渲染到纹理中,并创建一个用于后处理的缓冲区

        :param depthtex: 深度纹理,用于存储深度信息
        :param colortex: 颜色纹理,用于存储颜色信息
        :param auxtex: 辅助纹理,用于存储额外的渲染信息
        :param auxbits: 辅助位平面属性,用于配置渲染状态
        :param textures: 包含多种纹理的字典,用于指定颜色、深度和辅助纹理
        :param fbprops: 帧缓冲属性,用于配置缓冲区
        :param clamping: 是否禁用纹理的钳制
        :return: 用于后处理的四边形节点
        """

        # 从 textures 字典中提取纹理
        if (textures):
            colortex = textures.get("color", None)
            depthtex = textures.get("depth", None)
            auxtex = textures.get("aux", None)
            auxtex0 = textures.get("aux0", auxtex)
            auxtex1 = textures.get("aux1", None)
        else:
            auxtex0 = auxtex
            auxtex1 = None

        # 如果没有提供颜色纹理,则创建一个默认的颜色纹理
        if (colortex == None):
            colortex = Texture("filter-base-color")
            colortex.setWrapU(Texture.WMClamp)
            colortex.setWrapV(Texture.WMClamp)
        texgroup = (depthtex, colortex, auxtex0, auxtex1)

        # 选择离屏缓冲区的大小
        (winx, winy) = self.getScaledSize(1, 1, 1)
        if fbprops is not None:
            buffer = self.createBuffer("filter-base", winx, winy, texgroup, fbprops=fbprops)
        else:
            buffer = self.createBuffer("filter-base", winx, winy, texgroup)
        if (buffer == None):
            return None

        # 创建一个全屏四边形用于渲染
        cm = CardMaker("filter-base-quad")
        cm.setFrameFullscreenQuad()
        quad = NodePath(cm.generate())
        quad.setDepthTest(0)
        quad.setDepthWrite(0)
        quad.setTexture(colortex)
        quad.setColor(1, 0.5, 0.5, 1)

        # 配置摄像机状态
        cs = NodePath("dummy")
        cs.setState(self.camstate)
        # 是否需要启用 Shader Generator？
        # cs.setShaderAuto()
        if (auxbits):
            cs.setAttrib(AuxBitplaneAttrib.make(auxbits))
        if clamping is False:
            # 禁用 Shader Generator 中的钳制
            cs.setAttrib(LightRampAttrib.make_identity())
        self.camera.node().setInitialState(cs.getState())

        # 创建一个正交摄像机并附加到四边形
        quadcamnode = Camera("filter-quad-cam")
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)
        quadcamnode.setLens(lens)
        quadcam = quad.attachNewNode(quadcamnode)
        self.region.setCamera(quadcam)

        # 配置缓冲区清除状态
        self.setStackedClears(buffer, self.rclears, self.wclears)
        if (auxtex0):
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba0, 1)
            buffer.setClearValue(GraphicsOutput.RTPAuxRgba0, (0.5, 0.5, 1.0, 0.0))
        if (auxtex1):
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba1, 1)
        self.region.disableClears()
        if (self.isFullscreen()):
            self.win.disableClears()

        # 创建显示区域并设置摄像机
        dr = buffer.makeDisplayRegion()
        dr.disableClears()
        dr.setCamera(self.camera)
        dr.setActive(1)

        # 将缓冲区和尺寸添加到列表中
        self.buffers.append(buffer)
        self.sizes.append((1, 1, 1))
        return quad

    def createBuffer(self, name, xsize, ysize, texgroup, depthbits=1, fbprops=None):
        """
        重载 direct.filters.FilterManager.createBuffer 方法,用于创建渲染缓冲区

        :param name: 缓冲区的名称
        :param xsize: 缓冲区的宽度
        :param ysize: 缓冲区的高度
        :param texgroup: 包含深度纹理、颜色纹理和辅助纹理的组
        :param depthbits: 深度缓冲区的位数,默认为1
        :param fbprops: 帧缓冲区的属性
        :return: 创建的渲染缓冲区对象

        该方法通过设置窗口属性和帧缓冲属性来创建一个新的渲染缓冲区,并根据提供的纹理组配置缓冲区的渲染输出
        """
        # 设置窗口属性
        winprops = WindowProperties()
        winprops.setSize(xsize, ysize)
        props = FrameBufferProperties(FrameBufferProperties.getDefault())
        props.setBackBuffers(0)
        props.setRgbColor(1)

        # 创建输出缓冲区
        props.setDepthBits(depthbits)
        props.setStereo(self.win.isStereo())

        if fbprops is not None:
            props.addProperties(fbprops)
        depthtex, colortex, auxtex0, auxtex1 = texgroup

        if (auxtex0 != None):
            props.setAuxRgba(1)
        if (auxtex1 != None):
            props.setAuxRgba(2)
        buffer = base.graphicsEngine.makeOutput(
            self.win.getPipe(), name, -1,
            props, winprops, GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFResizeable,
            self.win.getGsg(), self.win)
        if (buffer == None):
            return buffer

        # 绑定颜色纹理
        if (colortex):
            buffer.addRenderTexture(colortex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPColor)

        # 绑定深度纹理
        if (depthtex):
            buffer.addRenderTexture(depthtex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPDepth)
        if (auxtex0):
            buffer.addRenderTexture(auxtex0, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba0)
        if (auxtex1):
            buffer.addRenderTexture(auxtex1, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba1)

        # 设置缓冲区排序和清除选项
        buffer.setSort(self.nextsort)
        buffer.disableClears()
        self.nextsort += 1
        return buffer
