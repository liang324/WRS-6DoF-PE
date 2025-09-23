import copy
import basis.data_adapter as da
import basis.trimesh_generator as trihelper
import modeling.model_collection as mc
import numpy as np
import open3d as o3d
from panda3d.core import NodePath, LineSegs, GeomNode, TransparencyAttrib, RenderModeAttrib
from visualization.panda.world import ShowBase
import warnings as wrn
import basis.robot_math as rm


class StaticGeometricModel(object):
    """
    加载对象作为静态几何模型 -> 不允许更改位置、旋转、颜色等属性,由于没有额外的元素,因此速度更快

    author: weiwei
    date: 20190312
    """

    def __init__(self, initor=None, name="defaultname", btransparency=True, btwosided=False):
        """
        初始化静态几何模型

        :param initor: 初始化器,可以是文件路径、trimesh 对象、NodePath 等
        :param name: 模型名称
        :param btransparency: 是否启用透明度
        :param btwosided: 是否启用双面渲染

        """
        if isinstance(initor, StaticGeometricModel):
            # 如果初始化器是另一个静态几何模型,则深拷贝其属性
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._name = copy.deepcopy(initor.name)
            self._localframe = copy.deepcopy(initor.localframe)
        else:
            # 创建一个祖父节点以分离装饰和原始节点
            self._name = name
            self._objpdnp = NodePath(name)
            if isinstance(initor, str):
                # 如果初始化器是字符串,假设是文件路径
                self._objpath = initor
                self._objtrm = da.trm.load(self._objpath)
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)

            elif isinstance(initor, da.trm.Trimesh):
                # 如果初始化器是 trimesh 对象
                self._objpath = None
                self._objtrm = initor
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm)
                objpdnp_raw.reparentTo(self._objpdnp)

            elif isinstance(initor, o3d.geometry.PointCloud):  # TODO pointcloud应该是pdnp或pdnp_raw
                # 如果初始化器是 open3d 点云
                self._objpath = None
                self._objtrm = da.trm.Trimesh(np.asarray(initor.points))
                objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)

            elif isinstance(initor, np.ndarray):  # TODO pointcloud应该是pdnp或pdnp_raw
                # 如果初始化器是 numpy 数组
                self._objpath = None
                if initor.shape[1] == 3:
                    # 处理形状为 nx3 的数组
                    self._objtrm = da.trm.Trimesh(initor)
                    # objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, np.array([1, 0, 0, 1]))
                    rgba_list = np.random.rand(len(self._objtrm.vertices), 4)
                    objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, rgba_list)

                elif initor.shape[1] == 7:
                    # 处理形状为 nx7 的数组
                    self._objtrm = da.trm.Trimesh(initor[:, :3])
                    objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, initor[:, 3:].tolist())
                    objpdnp_raw.setRenderMode(RenderModeAttrib.MPoint, 3)
                else:
                    # TODO depth UV?
                    # 未实现的其他形状
                    raise NotImplementedError
                objpdnp_raw.reparentTo(self._objpdnp)

            elif isinstance(initor, o3d.geometry.TriangleMesh):
                # 如果初始化器是 open3d 三角网格
                self._objpath = None
                self._objtrm = da.trm.Trimesh(vertices=initor.vertices, faces=initor.triangles,
                                              face_normals=initor.triangle_normals)
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)

            elif isinstance(initor, NodePath):
                # 如果初始化器是 NodePath
                self._objpath = None
                self._objtrm = None  # TODO: 将 NodePath 转换为 trimesh？
                objpdnp_raw = initor
                objpdnp_raw.reparentTo(self._objpdnp)

            else:
                # 默认情况
                self._objpath = None
                self._objtrm = None
                objpdnp_raw = NodePath("pdnp_raw")
                objpdnp_raw.reparentTo(self._objpdnp)

            if btransparency:
                # 设置透明度
                self._objpdnp.setTransparency(TransparencyAttrib.MDual)

            if btwosided:
                # 设置双面渲染
                self._objpdnp.getChild(0).setTwoSided(True)

            self._localframe = None

    @property
    def name(self):
        """
        只读属性,返回模型名称
        """
        return self._name

    @property
    def objpath(self):
        """
        只读属性,返回对象路径(文件路径)
        """
        return self._objpath

    @property
    def objpdnp(self):
        """
        只读属性,返回 Panda3D 的 NodePath 对象
        """
        return self._objpdnp

    @property
    def objpdnp_raw(self):
        """
        只读属性,返回原始 NodePath 对象
        """
        return self._objpdnp.getChild(0)

    @property
    def objtrm(self):
        """
        只读属性,返回 trimesh 对象

        20210328注释退出,允许无
        如果自我._objtrm为None:
        引发ValueError(“仅适用于具有trimesh的模型！
        """
        return self._objtrm

    @property
    def localframe(self):
        """
        只读属性,返回本地坐标系
        """
        return self._localframe

    @property
    def volume(self):
        """
        只读属性,返回模型体积

        如果模型没有 trimesh,则抛出异常
        """
        if self._objtrm is None:
            raise ValueError("仅适用于具有 trimesh 的模型！")
        return self._objtrm.volume

    def set_rgba(self, rgba):
        """
        设置模型的颜色

        :param rgba: 包含红、绿、蓝、透明度的列表或数组
        """
        self._objpdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def set_scale(self, scale=[1, 1, 1]):
        """
        设置模型的缩放比例

        :param scale: 包含 x, y, z 方向的缩放比例的列表或数组
        """
        self._objpdnp.setScale(scale[0], scale[1], scale[2])
        self._objtrm.apply_scale(scale)

    def set_vert_size(self, size=.005):
        """
        设置顶点的渲染大小

        :param size: 顶点大小
        """
        self.objpdnp_raw.setRenderModeThickness(size * 1000)

    def get_rgba(self):
        """
        获取模型的颜色

        :return: 包含红、绿、蓝、透明度的 NumPy 数组
        """
        return da.pdv4_to_npv4(self._objpdnp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_rgba(self):
        """
        清除模型的颜色设置
        """
        self._objpdnp.clearColor()

    def get_scale(self):
        """
        获取模型的缩放比例

        :return: 包含 x, y, z 方向的缩放比例的 NumPy 数组
        """
        return da.pdv3_to_npv3(self._objpdnp.getScale())

    def attach_to(self, obj):
        """
        将当前模型附加到指定对象上

        :param obj: 可以是 ShowBase、StaticGeometricModel、ModelCollection 或 NodePath
        """
        if isinstance(obj, ShowBase):
            # 附加到渲染环境
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):
            # 附加到另一个静态几何模型(用于装饰)
            self._objpdnp.reparentTo(obj.objpdnp)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_gm(self)  # 添加到模型集合
        elif isinstance(obj, NodePath):
            self._objpdnp.reparentTo(obj)  # 附加到 NodePath
        else:
            print(
                "必须是 ShowBase、StaticGeometricModel、GeometricModel、CollisionModel 或 CollisionModelCollection！")

    def detach(self):
        """
        从当前父节点分离模型
        """
        self._objpdnp.detachNode()

    def remove(self):
        """
        从场景中移除模型节点
        """
        self._objpdnp.removeNode()

    def show_localframe(self):
        """
        显示模型的本地坐标系
        """
        self._localframe = gen_frame()
        self._localframe.attach_to(self)

    def unshow_localframe(self):
        """
        隐藏模型的本地坐标系
        """
        if self._localframe is not None:
            self._localframe.remove()
            self._localframe = None

    def copy(self):
        """
        创建模型的深拷贝

        :return: 模型的副本
        """
        return copy.deepcopy(self)


class WireFrameModel(StaticGeometricModel):

    def __init__(self, initor=None, name="auto"):
        """
        初始化线框模型

        :param initor: 初始化器,可以是文件路径、trimesh 对象或 NodePath
        :param name: 模型名称

        """
        super().__init__(initor=initor, btransparency=False, name=name)
        self.objpdnp_raw.setRenderModeWireframe()  # 设置为线框模式
        self.objpdnp_raw.setLightOff()  # 关闭光照
        # self.set_rgba(rgba=[0,0,0,1])# 可选:设置颜色

    # 禁用某些函数
    def __getattr__(self, attr_name):
        """
        重写 __getattr__ 方法以禁用不支持的功能

        :param attr_name: 属性名称
        :raises AttributeError: 如果尝试访问不支持的功能
        """
        if attr_name == 'sample_surface':
            raise AttributeError("线框模型不支持采样表面！")
        return getattr(self._wrapped, attr_name)

    @property
    def name(self):
        """
        只读属性,返回模型名称
        """
        return self._name

    @property
    def objpath(self):
        """
        只读属性,返回对象路径
        """
        return self._objpath

    @property
    def objpdnp(self):
        """
        只读属性,返回对象的 Panda3D 节点路径
        """
        return self._objpdnp

    @property
    def objpdnp_raw(self):
        """
        只读属性,返回对象的原始 Panda3D 节点路径
        """
        return self._objpdnp.getChild(0)

    @property
    def objtrm(self):
        """
        只读属性,返回对象的 trimesh

        :raises ValueError: 如果对象没有 trimesh
        """
        if self._objtrm is None:
            raise ValueError("仅适用于具有 trimesh 的模型！")
        return self._objtrm

    @property
    def localframe(self):
        """
        只读属性,返回对象的本地坐标系
        """
        return self._localframe

    @property
    def volume(self):
        """
        只读属性,返回对象的体积

        :raises ValueError: 如果对象没有 trimesh
        """
        if self._objtrm is None:
            raise ValueError("仅适用于具有 trimesh 的模型！")
        return self._objtrm.volume

    def set_rgba(self, rgba):
        """
        设置对象的颜色(RGBA)

        :param rgba: RGBA颜色值
        :warning: 当前 WireFrame 实例的 set_rgba 函数未实现
        """
        wrn.warn("当前 WireFrame 实例的 set_rgba 函数未实现！")
        # self._objpdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def set_scale(self, scale=[1, 1, 1]):
        """
        设置对象的缩放比例

        :param scale: 缩放比例,默认为 [1, 1, 1]
        """
        self._objpdnp.setScale(scale[0], scale[1], scale[2])

    def get_rgba(self):
        """
        获取对象的颜色(RGBA)

        :return: RGBA颜色值
        """
        return da.pdv4_to_npv4(self._objpdnp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_rgba(self):
        """
        清除对象的颜色设置
        """
        self._objpdnp.clearColor()

    def get_scale(self):
        """
        获取对象的缩放比例

        :return: 缩放比例
        """
        return da.pdv3_to_npv3(self._objpdnp.getScale())

    def attach_to(self, obj):
        """
        将对象附加到指定的对象上

        :param obj: 可以是 ShowBase、StaticGeometricModel、ModelCollection 等
        :raises Exception: 如果对象类型不符合要求
        """
        if isinstance(obj, ShowBase):
            # 用于渲染到 base.render
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):  # 准备用于装饰,如本地坐标系
            self._objpdnp.reparentTo(obj.objpdnp)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_gm(self)
        else:
            raise Exception(
                "WRS 异常:必须是 ShowBase、StaticGeometricModel、GeometricModel、CollisionModel 或 CollisionModelCollection！")

    def detach(self):
        """
        从父节点分离对象的 Panda3D 节点路径
        """
        self._objpdnp.detachNode()

    def remove(self):
        """
        从场景中移除对象的 Panda3D 节点路径
        """
        self._objpdnp.removeNode()

    def show_localframe(self):
        """
        显示对象的本地坐标系
        """
        self._localframe = gen_frame()
        self._localframe.attach_to(self)

    def unshow_localframe(self):
        """
        隐藏对象的本地坐标系
        """
        if self._localframe is not None:
            self._localframe.removeNode()
            self._localframe = None


class GeometricModel(StaticGeometricModel):
    """
    加载对象作为几何模型

    该模型没有额外元素,因此速度更快
    author: weiwei
    date: 20190312
    """

    def __init__(self, initor=None, name="defaultname", btransparency=True, btwosided=False):
        """
        初始化 GeometricModel 对象

        :param initor: 初始化器,可以是文件路径、trimesh 对象或 Panda3D 的 NodePath
        :param name: 模型名称
        :param btransparency: 是否启用透明度
        :param btwosided: 是否启用双面渲染

        """
        if isinstance(initor, GeometricModel):
            # 深拷贝现有 GeometricModel 对象的属性
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._name = copy.deepcopy(initor.name)
            self._localframe = copy.deepcopy(initor.localframe)
        else:
            # 调用父类构造函数进行初始化
            super().__init__(initor=initor, name=name, btransparency=btransparency, btwosided=btwosided)
        # 自动设置着色器
        self.objpdnp_raw.setShaderAuto()

    def set_pos(self, npvec3):
        """
        设置对象的位置

        :param npvec3: 三维位置向量
        """
        self._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])

    def set_rotmat(self, npmat3):
        """
        设置对象的旋转矩阵

        :param npmat3: 3x3旋转矩阵
        """
        self._objpdnp.setQuat(da.npmat3_to_pdquat(npmat3))

    def set_homomat(self, npmat4):
        """
        设置对象的齐次变换矩阵

        :param npmat4: 4x4齐次变换矩阵
        """
        self._objpdnp.setPosQuat(da.npv3_to_pdv3(npmat4[:3, 3]), da.npmat3_to_pdquat(npmat4[:3, :3]))

    def set_rpy(self, roll, pitch, yaw):
        """
        使用欧拉角设置对象的姿态
        :param roll: 绕X轴的旋转角度(弧度)
        :param pitch: 绕Y轴的旋转角度(弧度)
        :param yaw: 绕Z轴的旋转角度(弧度)
        :return:
        author: weiwei
        date: 20190513
        """
        npmat3 = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self.set_rotmat(npmat3)

    def set_transparency(self, attribute):
        """
        设置对象的透明度属性

        :param attribute: 透明度属性
        :return: 设置结果
        """
        return self._objpdnp.setTransparency(attribute)

    def get_pos(self):
        """
        获取对象的位置

        :return: 三维位置向量
        """
        return da.pdv3_to_npv3(self._objpdnp.getPos())

    def get_rotmat(self):
        """
        获取对象的旋转矩阵

        :return: 3x3旋转矩阵
        """
        return da.pdquat_to_npmat3(self._objpdnp.getQuat())

    def get_homomat(self):
        """
        获取对象的齐次变换矩阵

        :return: 4x4齐次变换矩阵
        """
        npv3 = da.pdv3_to_npv3(self._objpdnp.getPos())
        npmat3 = da.pdquat_to_npmat3(self._objpdnp.getQuat())
        return rm.homomat_from_posrot(npv3, npmat3)

    def get_rpy(self):
        """
        获取对象的姿态(欧拉角表示)

        :return: [roll, pitch, yaw] 以弧度表示

        author: weiwei
        date: 20190513
        """
        npmat3 = self.get_rotmat()
        rpy = rm.rotmat_to_euler(npmat3, axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def sample_surface(self, radius=0.005, nsample=None, toggle_option='face_ids'):
        """
        从模型表面采样点

        :param radius: 采样半径
        :param nsample: 采样点数
        :param toggle_option: 返回附加信息的选项,'face_ids', 'normals', 或 None
        :return: 采样点和可选的附加信息

        author: weiwei
        date: 20191228
        """
        if self._objtrm is None:
            raise ValueError("仅适用于包含 trimesh 的模型！")
        if nsample is None:
            nsample = int(round(self.objtrm.area / ((radius * 0.3) ** 2)))
        points, face_ids = self.objtrm.sample_surface(nsample, radius=radius, toggle_faceid=True)
        points = rm.homomat_transform_points(self.get_homomat(), points)
        if toggle_option is None:
            return np.array(points)
        elif toggle_option == 'face_ids':
            return np.array(points), np.array(face_ids)
        elif toggle_option == 'normals':
            return np.array(points), rm.homomat_transform_points(self.get_homomat(), self.objtrm.face_normals[face_ids])
        else:
            print("toggle_option 必须是 None, 'face_ids', 或 'normals'")

    def copy(self):
        """
        创建对象的深拷贝

        :return: 对象的副本
        """
        return copy.deepcopy(self)


# 基元是固定的几何模型,一旦定义,就不能更改
# TODO: further 解耦 from Panda trimesh->staticgeometricmodel
def gen_linesegs(linesegs, thickness=0.001, rgba=[0, 0, 0, 1]):
    """
    生成线段集合的几何模型

    :param linesegs: 线段集合,格式为 [[pnt0, pnt1], [pnt0, pnt1], ...],其中 pnti 是 1x3 的 NumPy 数组,定义在局部坐标系中
    :param thickness: 线段的厚度,默认为 0.001
    :param rgba: 线段的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为 [0, 0, 0, 1](黑色,不透明)
    :return: 返回一个静态几何模型对象,表示生成的线段集合

    author: weiwei
    date: 20161216, 20201116
    """
    M_TO_PIXEL = 3779.53  # 将米转换为像素的比例
    ls = LineSegs()  # 创建线段对象
    ls.setThickness(thickness * M_TO_PIXEL)  # 设置线段厚度
    ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])  # 设置线段颜色和透明度
    for p0p1tuple in linesegs:
        ls.moveTo(p0p1tuple[0][0], p0p1tuple[0][1], p0p1tuple[0][2])  # 移动到线段起点
        ls.drawTo(p0p1tuple[1][0], p0p1tuple[1][1], p0p1tuple[1][2])  # 绘制到线段终点
    lsnp = NodePath(ls.create())  # 创建节点路径
    lsnp.setTransparency(TransparencyAttrib.MDual)  # 设置透明度属性
    lsnp.setLightOff()  # 关闭光照影响
    ls_sgm = StaticGeometricModel(lsnp)  # 创建静态几何模型
    return ls_sgm


# def gen_linesegs(verts, thickness=0.005, rgba=[0,0,0,1]):
#     """
#     gen continuous linsegs
#     :param verts: nx3 list, each nearby pair will be used to draw one segment, defined in a local 0 frame
#     :param rgba:
#     :param thickness:
#     :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
#     :return: a geomtric model
#     author: weiwei
#     date: 20161216
#     """
#     segs = LineSegs()
#     segs.setThickness(thickness * 1000.0)
#     segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
#     for i in range(len(verts) - 1):
#         tmpstartvert = verts[i]
#         tmpendvert = verts[i + 1]
#         segs.moveTo(tmpstartvert[0], tmpstartvert[1], tmpstartvert[2])
#         segs.drawTo(tmpendvert[0], tmpendvert[1], tmpendvert[2])
#     lsnp = NodePath('linesegs')
#     lsnp.attachNewNode(segs.create())
#     lsnp.setTransparency(TransparencyAttrib.MDual)
#     ls_sgm = StaticGeometricModel(lsnp)
#     return ls_sgm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1], subdivisions=3):
    """
    生成一个球体的几何模型
    :param pos: 球体的中心位置,默认为原点 [0, 0, 0]
    :param radius: 球体的半径,默认为 0.01
    :param rgba: 球体的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为 [1, 0, 0, 1](红色,不透明)
    :param subdivisions: 球体的细分级别,默认为 3,细分级别越高,球体越平滑
    :return: 返回一个静态几何模型对象,表示生成的球体
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trihelper.gen_sphere(pos, radius, subdivisions)  # 生成球体的三角形网格
    sphere_sgm = StaticGeometricModel(sphere_trm)  # 创建静态几何模型对象
    sphere_sgm.set_rgba(rgba)
    return sphere_sgm


def gen_ellipsoid(pos=np.array([0, 0, 0]),
                  axmat=np.eye(3),
                  rgba=[1, 1, 0, .3]):
    """
    生成椭球体的几何模型

    :param pos: 椭球体的中心位置,默认为原点 [0, 0, 0]
    :param axmat: 3x3 矩阵,每列表示椭球体的一个轴,默认为单位矩阵
    :param rgba: 椭球体的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为黄色 [1, 1, 0, .3]
    :return: 返回一个静态几何模型对象,表示生成的椭球体

    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trihelper.gen_ellipsoid(pos=pos, axmat=axmat)
    ellipsoid_sgm = StaticGeometricModel(ellipsoid_trm)
    ellipsoid_sgm.set_rgba(rgba)
    return ellipsoid_sgm


def gen_stick(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              thickness=.005, type="rect",
              rgba=[1, 0, 0, 1], sections=8):
    """
    生成棍状物体的几何模型

    :param spos: 棍状物体的起始位置,默认为原点 [0, 0, 0]
    :param epos: 棍状物体的结束位置,默认值为 [0.1, 0, 0]
    :param thickness: 棍状物体的厚度,默认为 0.005
    :param type: 棍状物体的类型,可以是 "rect" 或 "round",默认为 "rect"
    :param rgba: 棍状物体的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :param sections: 棍状物体的截面数,默认为 8
    :return: 返回一个静态几何模型对象,表示生成的棍状物体

    author: weiwei
    date: 20191229osaka
    """
    stick_trm = trihelper.gen_stick(spos=spos, epos=epos, thickness=thickness, type=type, sections=sections)
    stick_sgm = StaticGeometricModel(stick_trm)
    stick_sgm.set_rgba(rgba)
    return stick_sgm


def gen_dashstick(spos=np.array([0, 0, 0]),
                  epos=np.array([.1, 0, 0]),
                  thickness=.005,
                  lsolid=None,
                  lspace=None,
                  rgba=[1, 0, 0, 1],
                  type="rect"):
    """
    生成虚线棍状物体的几何模型

    :param spos: 虚线的起始位置,默认值为原点 [0, 0, 0]
    :param epos: 虚线的结束位置,默认值为 [0.1, 0, 0]
    :param thickness: 虚线的厚度,默认值为 0.005
    :param lsolid: 虚线中实心部分的长度.如果未提供,默认值为 1 * thickness
    :param lspace: 虚线中空心部分的长度.如果未提供,默认值为 1.5 * thickness
    :param rgba: 虚线的颜色,采用 RGBA 格式,默认值为红色 [1, 0, 0, 1]
    :param type: 虚线的类型,默认值为 "rect"
    :return: 返回一个静态几何模型对象,表示生成的虚线棍状物体

    author: weiwei
    date: 20200625osaka
    """
    dashstick_trm = trihelper.gen_dashstick(spos=spos,
                                            epos=epos,
                                            lsolid=lsolid,
                                            lspace=lspace,
                                            thickness=thickness,
                                            sticktype=type)
    dashstick_sgm = StaticGeometricModel(dashstick_trm)
    dashstick_sgm.set_rgba(rgba=rgba)
    return dashstick_sgm


def movegmbox(extent=np.array([1, 1, 1]),
              homomat=np.eye(4),
              pos=np.array([0, 0, 0]),
              rgba=[1, 0, 0, 1]):
    """
    生成一个可移动的盒子几何模型

    :param homomat: 盒子的变换矩阵,默认为单位矩阵
    :param pos: 盒子的初始位置,默认为原点 [0, 0, 0]
    :param rgba: 盒子的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :return: 返回一个几何模型对象,表示生成的盒子
    """
    pos = pos
    box = trihelper.gen_box(extent=extent, homomat=homomat)
    box_gm = GeometricModel(box)
    box_gm.set_pos(pos)
    box_gm.set_rgba(rgba=rgba)
    return box_gm


def gen_box(extent=np.array([.1, .1, .1]),
            homomat=np.eye(4),
            rgba=[1, 0, 0, 1]):
    """
    生成一个盒子几何模型

    :param extent: 盒子的尺寸,默认为 [0.1, 0.1, 0.1]
    :param homomat: 盒子的变换矩阵,默认为单位矩阵
    :param rgba: 盒子的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :return: 返回一个静态几何模型对象,表示生成的盒子

    author: weiwei
    date: 20191229osaka
    """
    box_trm = trihelper.gen_box(extent=extent, homomat=homomat)
    box_sgm = StaticGeometricModel(box_trm)
    box_sgm.set_rgba(rgba=rgba)
    return box_sgm


def gen_cylinder(radius=0.1, height=0.2, section=100, homomat=np.eye(4), rgba=(1, 1, 0, 1)):
    """
    生成一个圆柱体几何模型

    :param radius: 圆柱体的半径,默认为 0.1
    :param height: 圆柱体的高度,默认为 0.2
    :param section: 圆柱体的截面数,默认为 100
    :param homomat: 圆柱体的变换矩阵,默认为单位矩阵
    :param rgba: 圆柱体的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为黄色 [1, 1, 0, 1]
    :return: 返回一个几何模型对象,表示生成的圆柱体

    author: hu
    date: 20220113
    """
    cld_trm = trihelper.gen_cylinder(radius=radius, height=height, section=section, homomat=homomat)
    cld_sgm = StaticGeometricModel(cld_trm)
    cld_sgm = GeometricModel(cld_trm)
    cld_sgm.set_rgba(rgba=rgba)
    return cld_sgm


def gen_capsule(spos=(0, 0, 0), epos=(0, 0, 0.01), section=[100, 100], radius=0.005, rgba=(1, 1, 0, 1)):
    """
    生成一个胶囊体几何模型

    :param spos: 胶囊体的起始位置,默认为 (0, 0, 0)
    :param epos: 胶囊体的结束位置,默认为 (0, 0, 0.01)
    :param section: 胶囊体的截面数,默认为 [100, 100]
    :param radius: 胶囊体的半径,默认为 0.005
    :param rgba: 胶囊体的颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为黄色 [1, 1, 0, 1]
    :return: 返回一个静态几何模型对象,表示生成的胶囊体

    author: hu
    date: 20220113
    """
    cld_trm = trihelper.gen_roundstick(spos=spos, epos=epos, radius=radius, count=section)
    # cld_trm = trihelper.gen_capsule(spos=spos, epos= epos, section = section, homomat= homomat)
    cld_sgm = StaticGeometricModel(cld_trm)
    cld_sgm.set_rgba(rgba=rgba)
    return cld_sgm


def gen_dumbbell(spos=np.array([0, 0, 0]),
                 epos=np.array([.1, 0, 0]),
                 thickness=.005,
                 rgba=[1, 0, 0, 1]):
    """
    生成一个哑铃形状的几何模型

    :param spos: 起始位置,默认为 [0, 0, 0]
    :param epos: 结束位置,默认为 [0.1, 0, 0]
    :param thickness: 哑铃的厚度,默认为 0.005
    :param rgba: 颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :return: 返回一个静态几何模型对象,表示生成的哑铃

    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    dumbbell_trm = trihelper.gen_dumbbell(spos=spos, epos=epos, thickness=thickness)
    dumbbell_sgm = StaticGeometricModel(dumbbell_trm)
    dumbbell_sgm.set_rgba(rgba=rgba)
    return dumbbell_sgm


def gen_cone(spos=np.array([0, 0, 0]),
             epos=np.array([0.1, 0, 0]),
             rgba=np.array([.7, .7, .7, .3]),
             radius=0.005,
             sections=8):
    """
    生成一个圆锥形状的几何模型

    :param spos: 起始位置,默认为 [0, 0, 0]
    :param epos: 结束位置,默认为 [0.1, 0, 0]
    :param radius: 圆锥的底面半径,默认为 0.005
    :param sections: 圆锥的截面数,默认为 8
    :param rgba: 颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为灰色 [0.7, 0.7, 0.7, 0.3]
    :return: 返回一个几何模型对象,表示生成的圆锥

    author: weiwei
    date: 20210625
    """
    cone_trm = trihelper.gen_cone(spos=spos, epos=epos, radius=radius, sections=sections)
    cone_sgm = GeometricModel(cone_trm)
    cone_sgm.set_rgba(rgba=rgba)
    return cone_sgm


def gen_section(spos=np.array([0, 0, 0]),
                epos=np.array([0.1, 0, 0]),
                rgba=np.array([.7, .7, .7, .3]),
                height_vec=np.array([0, 0, 1]), height=0.01, angle=30, section=8):
    """
    生成一个截面形状的几何模型

    :param spos: 起始位置,默认为 [0, 0, 0]
    :param epos: 结束位置,默认为 [0.1, 0, 0]
    :param height_vec: 高度方向向量,默认为 [0, 0, 1]
    :param height: 截面的高度,默认为 0.01
    :param angle: 截面的角度,默认为 30 度
    :param section: 截面的截面数,默认为 8
    :param rgba: 颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为灰色 [0.7, 0.7, 0.7, 0.3]
    :return: 返回一个几何模型对象,表示生成的截面

    author: hu
    date: 20240611
    """
    height_vec = rm.unit_vector(height_vec)
    # print(height_vec)
    section_trm = trihelper.gen_section(spos=spos, epos=epos, height_vec=height_vec, height=height, angle=angle,
                                        section=section)
    # print(section_trm.faces)
    section_sgm = GeometricModel(section_trm)
    section_sgm.set_rgba(rgba=rgba)
    return section_sgm


def gen_arrow(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              thickness=.005, rgba=[1, 0, 0, 1],
              type="rect"):
    """
    生成一个箭头形状的几何模型

    :param spos: 起始位置,默认为 [0, 0, 0]
    :param epos: 结束位置,默认为 [0.1, 0, 0]
    :param thickness: 箭头的厚度,默认为 0.005
    :param rgba: 颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :param type: 箭头的类型,默认为 "rect"
    :return: 返回一个静态几何模型对象,表示生成的箭头

    author: weiwei
    date: 20200115osaka
    """
    arrow_trm = trihelper.gen_arrow(spos=spos, epos=epos, thickness=thickness, sticktype=type)
    arrow_sgm = StaticGeometricModel(arrow_trm)
    arrow_sgm.set_rgba(rgba=rgba)
    return arrow_sgm


def gen_dasharrow(spos=np.array([0, 0, 0]),
                  epos=np.array([.1, 0, 0]),
                  thickness=.005, lsolid=None,
                  lspace=None,
                  rgba=[1, 0, 0, 1], type="rect"):
    """
    生成一个虚线箭头形状的几何模型

    :param spos: 起始位置,默认为 [0, 0, 0]
    :param epos: 结束位置,默认为 [0.1, 0, 0]
    :param thickness: 箭头的厚度,默认为 0.005
    :param lsolid: 实线部分的长度,默认为厚度的 1 倍
    :param lspace: 空白部分的长度,默认为厚度的 1.5 倍
    :param rgba: 颜色和透明度,格式为 [红, 绿, 蓝, 透明度],默认为红色 [1, 0, 0, 1]
    :param type: 箭头的类型,默认为 "rect"
    :return: 返回一个静态几何模型对象,表示生成的虚线箭头

    author: weiwei
    date: 20200625osaka
    """
    dasharrow_trm = trihelper.gen_dasharrow(spos=spos,
                                            epos=epos,
                                            lsolid=lsolid,
                                            lspace=lspace,
                                            thickness=thickness,
                                            sticktype=type)
    dasharrow_sgm = StaticGeometricModel(dasharrow_trm)
    dasharrow_sgm.set_rgba(rgba=rgba)
    return dasharrow_sgm


def gen_frame(pos=np.array([0, 0, 0]),
              rotmat=np.eye(3),
              length=.1,
              thickness=.005,
              rgbmatrix=None,
              alpha=None,
              plotname="frame"):
    """
    生成一个坐标轴框架,用于附加

    :param pos: 坐标轴的起始位置,默认为 [0, 0, 0]
    :param rotmat: 旋转矩阵,用于定义坐标轴的方向,默认为单位矩阵
    :param length: 坐标轴的长度,默认为 0.1
    :param thickness: 坐标轴的厚度,默认为 0.005
    :param rgbmatrix: 每列表示每个基轴的颜色,默认为红、绿、蓝
    :param alpha: 透明度,默认为 1
    :param plotname: 绘图名称,默认为 "frame"
    :return: 返回一个静态几何模型对象,表示生成的坐标轴框架

    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    endx = pos + rotmat[:, 0] * length
    endy = pos + rotmat[:, 1] * length
    endz = pos + rotmat[:, 2] * length

    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]

    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha

    # TODO 20201202 change it to StaticGeometricModelCollection
    frame_nodepath = NodePath(plotname)

    arrowx_trm = trihelper.gen_arrow(spos=pos, epos=endx, thickness=thickness)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)

    arrowy_trm = trihelper.gen_arrow(spos=pos, epos=endy, thickness=thickness)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)

    arrowz_trm = trihelper.gen_arrow(spos=pos, epos=endz, thickness=thickness)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)

    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)

    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_mycframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, alpha=None, plotname="mycframe"):
    """
    创建一个连接轴,用磁铁为x,黄色为y,青色为z

    :param pos: 坐标轴的起始位置,默认为 [0, 0, 0]
    :param rotmat: 旋转矩阵,用于定义坐标轴的方向,默认为单位矩阵
    :param length: 坐标轴的长度,默认为 0.1
    :param thickness: 坐标轴的厚度,默认为 0.005
    :param alpha: 透明度,默认为 1
    :param plotname: 绘图名称,默认为 "mycframe"
    :return: 返回一个静态几何模型对象,表示生成的坐标轴框架

    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    rgbmatrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]]).T
    return gen_frame(pos=pos, rotmat=rotmat, length=length, thickness=thickness, rgbmatrix=rgbmatrix, alpha=alpha,
                     plotname=plotname)


def gen_dashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, lsolid=None, lspace=None,
                  rgbmatrix=None, alpha=None, plotname="dashframe"):
    """
    生成一个虚线坐标轴框架,用于附加

    :param pos: 坐标轴的起始位置,默认为 [0, 0, 0]
    :param rotmat: 旋转矩阵,用于定义坐标轴的方向,默认为单位矩阵
    :param length: 坐标轴的长度,默认为 0.1
    :param thickness: 坐标轴的厚度,默认为 0.005
    :param lsolid: 实线部分的长度,默认为厚度的 1 倍
    :param lspace: 空白部分的长度,默认为厚度的 1.5 倍
    :param rgbmatrix: 每列表示每个基轴的颜色,默认为红、绿、蓝
    :param alpha: 透明度,默认为 1
    :param plotname: 绘图名称,默认为 "dashframe"
    :return: 返回一个静态几何模型对象,表示生成的虚线坐标轴框架

    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * length
    endy = pos + rotmat[:, 1] * length
    endz = pos + rotmat[:, 2] * length
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # TODO 20201202 change it to StaticGeometricModelCollection
    frame_nodepath = NodePath(plotname)
    arrowx_trm = trihelper.gen_dasharrow(spos=pos, epos=endx, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)

    arrowy_trm = trihelper.gen_dasharrow(spos=pos, epos=endy, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)

    arrowz_trm = trihelper.gen_dasharrow(spos=pos, epos=endz, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              radius=.005,
              thickness=.0015,
              rgba=[1, 0, 0, 1],
              sections=8,
              discretization=24):
    """
    生成一个圆环(环面)

    :param axis: 圆环旋转的轴,1x3 的 numpy 数组
    :param starting_vector: 起始向量,用于定义圆环的起始方向
    :param portion: 圆环的部分比例,范围为 0.0 到 1.0
    :param center: 圆环的中心位置,1x3 的 numpy 数组
    :param radius: 圆环的半径
    :param thickness: 圆环的厚度
    :param rgba: 圆环的颜色和透明度,格式为 [R, G, B, A]
    :param sections: 圆环的截面数
    :param discretization: 圆环的离散化程度
    :return: 返回一个静态几何模型对象,表示生成的圆环

    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gen_torus(axis=axis,
                                    starting_vector=starting_vector,
                                    portion=portion,
                                    center=center,
                                    radius=radius,
                                    thickness=thickness,
                                    sections=sections,
                                    discretization=discretization)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_curveline(pseq, r, section=5, toggledebug=False):
    """
    生成一条曲线

    :param pseq: 点序列,定义曲线的路径
    :param r: 曲线的半径
    :param section: 曲线的截面数
    :param toggledebug: 是否开启调试模式
    :return: 返回一个几何模型对象,表示生成的曲线
    """

    def get_rotseq_by_pseq(pseq):
        rotseq = []
        pre_n = None
        for i in range(1, len(pseq) - 1):
            v1 = pseq[i - 1] - pseq[i]
            v2 = pseq[i] - pseq[i + 1]
            n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
            if pre_n is not None:
                if rm.angle_between_vectors(n, pre_n) > np.pi / 2:
                    n = -n
            x = np.cross(v1, n)
            rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
            rotseq.append(rot)
            pre_n = n
        rotseq = [rotseq[0]] + rotseq + [rotseq[-1]]
        return rotseq

    rotseq = get_rotseq_by_pseq(pseq)
    # gen_sphere(pseq[0], radius=0.0002, rgba=[0, 1, 0, 1]).attach_to(base)
    return GeometricModel(trihelper.gen_curveline(pseq, rotseq, r, section, toggledebug))


def gen_ellipse(center, points, r, section, toggledebug=False):
    """
    生成一个椭圆

    :param center: 椭圆的中心
    :param points: 定义椭圆的三个点
    :param r: 椭圆的半径
    :param section: 椭圆的截面数
    :param toggledebug: 是否开启调试模式
    :return: 返回一个几何模型对象,表示生成的椭圆
    """
    a = np.linalg.norm(points[0] - center)
    b = np.linalg.norm(points[1] - center)
    import hu.humath as hm
    surface = hm.getsurfacefrom3pnt(points[:3])
    normal = np.asarray(surface[:3])
    rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), normal)
    pos = center
    homomat = rm.homomat_from_posrot(pos, rotmat)

    def ellipse_curve(a=0.01, b=0.02, homomat=np.eye(4)):
        disc = 50
        theta_list = np.linspace(0, 2 * np.pi * (disc + 1) / disc, disc + 1)
        xy_list = []
        for theta in theta_list:
            r = (a * b) / (np.sqrt((b * b * np.cos(theta) * np.cos(theta)) + (a * a * np.sin(theta) * np.sin(theta))))
            coordinat = rm.homomat_transform_points(homomat, np.array([r * np.cos(theta), r * np.sin(theta), 0]))
            xy_list.append(coordinat)
        return xy_list

    curve = ellipse_curve(a, b, homomat)

    return gen_curveline(curve, r, section=section, toggledebug=False)


def gen_halfellipse(points, r, section, toggledebug=False):
    """
    生成一个半椭圆

    :param points: 定义半椭圆的三个点
    :param r: 半椭圆的半径
    :param section: 半椭圆的截面数
    :param toggledebug: 是否开启调试模式
    :return: 返回一个几何模型对象,表示生成的半椭圆
    """
    center = 0.5 * (points[0] + points[2])

    a = np.linalg.norm(points[0] - center)
    b = np.linalg.norm(points[1] - center)
    import hu.humath as hm
    surface = hm.getsurfacefrom3pnt(points[:3])
    normal = np.asarray(surface[:3])
    a_unit = rm.unit_vector(points[0] - center)
    b_unit = rm.unit_vector(points[1] - center)
    rotmat = np.array([[a_unit[0], b_unit[0], normal[0]],
                       [a_unit[1], b_unit[1], normal[1]],
                       [a_unit[2], b_unit[2], normal[2]]])
    # rotmat = rm.rotmat_between_vectors(np.array([0,0,1]), normal)
    pos = center
    # pos = 0.5*(points[0]-points[2])
    homomat = rm.homomat_from_posrot(pos, rotmat)

    def ellipse_curve(a=0.01, b=0.02, homomat=np.eye(4)):
        disc = 50
        theta_list = np.linspace(0, 1 * np.pi * (disc + 1) / disc, disc + 1)
        xy_list = []
        for theta in theta_list:
            r = (a * b) / (np.sqrt((b * b * np.cos(theta) * np.cos(theta)) + (a * a * np.sin(theta) * np.sin(theta))))
            coordinat = rm.homomat_transform_points(homomat, np.array([r * np.cos(theta), r * np.sin(theta), 0]))
            xy_list.append(coordinat)
        return xy_list

    curve = ellipse_curve(a, b, homomat)

    return gen_curveline(curve, r, section=section, toggledebug=False)


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=0.1,
                  thickness=0.005,
                  rgba=[1, 0, 0, 1],
                  lsolid=None,
                  lspace=None,
                  sections=8,
                  discretization=24):
    """
    生成一个虚线圆环(环面)

    :param axis: 圆环旋转的轴,1x3 的 numpy 数组
    :param portion: 圆环的部分比例,范围为 0.0 到 1.0
    :param center: 圆环的中心位置,1x3 的 numpy 数组
    :param radius: 圆环的半径
    :param thickness: 圆环的厚度
    :param rgba: 圆环的颜色和透明度,格式为 [R, G, B, A]
    :param lsolid: 实线部分的长度
    :param lspace: 空白部分的长度
    :param sections: 圆环的截面数
    :param discretization: 圆环的离散化程度
    :return: 返回一个静态几何模型对象,表示生成的虚线圆环

    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gen_dashtorus(axis=axis,
                                        portion=portion,
                                        center=center,
                                        radius=radius,
                                        thickness=thickness,
                                        lsolid=lsolid,
                                        lspace=lspace,
                                        sections=sections,
                                        discretization=discretization)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=.05,
                  thickness=.005,
                  rgba=[1, 0, 0, 1],
                  sections=8,
                  discretization=24):
    """
    生成一个圆形箭头

    :param axis: 圆形箭头旋转的轴,1x3 的 numpy 数组
    :param portion: 圆形箭头的部分比例,范围为 0.0 到 1.0
    :param center: 圆形箭头的中心位置,1x3 的 numpy 数组
    :param radius: 圆形箭头的半径
    :param thickness: 圆形箭头的厚度
    :param rgba: 圆形箭头的颜色和透明度,格式为 [R, G, B, A]
    :param sections: 圆形箭头的截面数
    :param discretization: 圆形箭头的离散化程度
    :return: 返回一个静态几何模型对象,表示生成的圆形箭头

    author: weiwei
    date: 20200602
    """
    circarrow_trm = trihelper.gen_circarrow(axis=axis,
                                            starting_vector=starting_vector,
                                            portion=portion,
                                            center=center,
                                            radius=radius,
                                            thickness=thickness,
                                            sections=sections,
                                            discretization=discretization)
    circarrow_sgm = StaticGeometricModel(circarrow_trm)
    circarrow_sgm.set_rgba(rgba=rgba)
    return circarrow_sgm


def gen_pointcloud(points, rgbas=[[0, 0, 0, .7]], pntsize=3):
    """
    生成点云,不要直接使用这个原始函数
    使用 environment.collisionmodel 来调用它

    :param points: nx3 列表,表示点的坐标
    :param rgbas: 颜色和透明度列表；如果指定多个颜色,需与每个点对应；默认为统一颜色
    :param pntsize: 点的大小
    :return: 返回静态几何模型
    """
    # 从点和颜色信息生成节点路径
    pointcloud_nodepath = da.nodepath_from_points(points, rgbas)
    # 设置渲染模式为点,并指定点的大小
    pointcloud_nodepath.setRenderMode(RenderModeAttrib.MPoint, pntsize)
    # 创建静态几何模型
    pointcloud_sgm = StaticGeometricModel(pointcloud_nodepath)
    # 返回生成的静态几何模型
    return pointcloud_sgm

def gen_rgb_pointcloud(rgb_points, pntsize=3):
    """
    生成 RGB 点云的静态几何模型

    请勿直接使用此原始函数,建议通过 environment.collisionmodel 调用

    :param rgb_points: 包含 RGB 信息的点云数据,形状为 nx6 的数组,其中前 3 列为点的坐标,后 3 列为 RGB 颜色值
    :param pntsize: 点的大小,默认为 3
    :return: 返回一个包含点云的静态几何模型对象
    """
    points = rgb_points[:, :3]
    rgbs = rgb_points[:, 3:]
    rgbas = np.append(rgbs, np.ones((len(rgbs), 1)), axis=1)
    pointcloud_nodepath = da.nodepath_from_points(points, rgbas)
    pointcloud_nodepath.setRenderMode(RenderModeAttrib.MPoint, pntsize)
    pointcloud_sgm = StaticGeometricModel(pointcloud_nodepath)
    return pointcloud_sgm

def gen_submesh(verts, faces, rgba=[1, 0, 0, 1]):
    """
    生成子网格

    TODO 20201202: replace pandanode with trimesh
    :param verts: 顶点数组,格式为 np.array([[v00, v01, v02], [v10, v11, v12], ...])
    :param faces: 面数组,格式为 np.array([[ti00, ti01, ti02], [ti10, ti11, ti12], ...])
    :param rgba: 子网格的颜色和透明度,格式为 [R, G, B, A]
    :return: 返回静态几何模型

    author: weiwei
    date: 20171219
    """
    # 生成顶点法线
    vertnormals = np.zeros((len(verts), 3))
    for fc in faces:
        vert0 = verts[fc[0], :]
        vert1 = verts[fc[1], :]
        vert2 = verts[fc[2], :]
        facenormal = np.cross(vert2 - vert1, vert0 - vert1)
        vertnormals[fc[0], :] = vertnormals[fc[0]] + facenormal
        vertnormals[fc[1], :] = vertnormals[fc[1]] + facenormal
        vertnormals[fc[2], :] = vertnormals[fc[2]] + facenormal
    for i in range(0, len(vertnormals)):
        vertnormals[i, :] = vertnormals[i, :] / np.linalg.norm(vertnormals[i, :])
    geom = da.pandageom_from_vvnf(verts, vertnormals, faces)
    node = GeomNode('surface')
    node.addGeom(geom)
    surface_nodepath = NodePath('surface')
    surface_nodepath.attachNewNode(node)
    surface_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    surface_nodepath.setTransparency(TransparencyAttrib.MDual)
    surface_nodepath.setTwoSided(True)
    surface_sgm = StaticGeometricModel(surface_nodepath)
    return surface_sgm


def gen_polygon(verts, thickness=0.002, rgba=[0, 0, 0, .7]):
    """
    生成一个多边形的静态几何模型

    :param verts: 顶点列表,格式为 [[x0, y0, z0], [x1, y1, z1], ...]
    :param thickness: 线段的厚度
    :param rgba: 多边形的颜色和透明度,格式为 [R, G, B, A]
    :return: 返回生成的静态几何模型对象

    author: weiwei
    date: 20201115
    """
    segs = LineSegs()
    segs.setThickness(thickness)
    segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    for i in range(len(verts) - 1):
        segs.moveTo(verts[i][0], verts[i][1], verts[i][2])
        segs.drawTo(verts[i + 1][0], verts[i + 1][1], verts[i + 1][2])
    polygon_nodepath = NodePath('polygons')
    polygon_nodepath.attachNewNode(segs.create())
    polygon_nodepath.setTransparency(TransparencyAttrib.MDual)
    polygon_sgm = StaticGeometricModel(polygon_nodepath)
    return polygon_sgm


def gen_frame_box(extent=[.02, .02, .02], homomat=np.eye(4), rgba=[0, 0, 0, 1], thickness=.001):
    """
    绘制一个3D框,仅显示边缘
    :param extent: 框的尺寸,格式为 [x_extent, y_extent, z_extent]
    :param homomat: 变换矩阵,4x4的numpy数组
    :param rgba: 框的颜色和透明度,格式为 [R, G, B, A]
    :param thickness: 边缘的厚度
    :return: 返回生成的静态几何模型对象
    """
    M_TO_PIXEL = 3779.53
    ls = LineSegs()
    ls.setThickness(thickness * M_TO_PIXEL)
    ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    center_pos = homomat[:3, 3]
    x_axis = homomat[:3, 0]
    y_axis = homomat[:3, 1]
    z_axis = homomat[:3, 2]
    x_min, x_max = -x_axis * extent[0] / 2, x_axis * extent[0] / 2
    y_min, y_max = -y_axis * extent[1] / 2, y_axis * extent[1] / 2
    z_min, z_max = -z_axis * extent[2] / 2, z_axis * extent[2] / 2
    # max, max, max
    print(center_pos + np.array([x_max, y_max, z_max]))
    ls.moveTo(da.npv3_to_pdv3(center_pos + x_max + y_max + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_max + y_max + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_max + y_min + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_max + y_min + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_max + y_max + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_max + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_min + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_min + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_max + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_max + z_max))
    ls.moveTo(da.npv3_to_pdv3(center_pos + x_max + y_max + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_max + z_min))
    ls.moveTo(da.npv3_to_pdv3(center_pos + x_max + y_min + z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_min + z_min))
    ls.moveTo(da.npv3_to_pdv3(center_pos + x_max + y_min + z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos + x_min + y_min + z_max))
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def gen_surface(surface_callback, rng, granularity=.01):
    """
    生成一个表面几何模型

    :param surface_callback: 用于生成表面的回调函数
    :param rng: 范围参数,用于定义表面生成的范围
    :param granularity: 表面的细粒度
    :return: 返回生成的几何模型对象
    """
    surface_trm = trihelper.gen_surface(surface_callback, rng, granularity)
    surface_gm = GeometricModel(surface_trm, btwosided=True)
    return surface_gm


if __name__ == "__main__":
    import os
    import math
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnygm = GeometricModel(objpath)
    bunnygm.set_rgba([0.7, 0.7, 0.0, 1.0])
    bunnygm.attach_to(base)
    bunnygm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnygm.set_rotmat(rotmat)
    # base.run()

    bunnygm1 = bunnygm.copy()
    bunnygm1.set_rgba([0.7, 0, 0.7, 1.0])
    bunnygm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.set_pos(np.array([0, .01, 0]))
    bunnygm1.set_rotmat(rotmat)

    bunnygm2 = bunnygm1.copy()
    bunnygm2.set_rgba([0, 0.7, 0.7, 1.0])
    bunnygm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm2.set_pos(np.array([0, .2, 0]))
    bunnygm2.set_rotmat(rotmat)
    bunnygm2.set_scale([2, 1, 3])
    # base.run()

    bunnygmpoints, _ = bunnygm.sample_surface()
    bpgm = GeometricModel(bunnygmpoints)
    bunnygm1points, _ = bunnygm1.sample_surface()
    bpgm1 = GeometricModel(bunnygm1points)
    bunnygm2points, _ = bunnygm2.sample_surface()
    bpgm2 = GeometricModel(bunnygm2points)
    bpgm.attach_to(base)
    bpgm.set_scale([2, 1, 3])
    bpgm.set_vert_size(.01)
    bpgm1.attach_to(base)
    bpgm2.attach_to(base)
    # base.run()

    lsgm = gen_linesegs([[np.array([.1, 0, .01]), np.array([.01, 0, .01])],
                         [np.array([.01, 0, .01]), np.array([.1, 0, .1])],
                         [np.array([.1, 0, .1]), np.array([.1, 0, .01])]])
    lsgm.attach_to(base)

    gen_circarrow(radius=.1, portion=.8).attach_to(base)
    gen_dasharrow(spos=np.array([0, 0, 0]), epos=np.array([0, 0, 2])).attach_to(base)
    gen_dashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3)).attach_to(base)
    axmat = rm.rotmat_from_axangle([1, 1, 1], math.pi / 4)
    gen_frame(rotmat=axmat).attach_to(base)
    axmat[:, 0] = .1 * axmat[:, 0]
    axmat[:, 1] = .07 * axmat[:, 1]
    axmat[:, 2] = .3 * axmat[:, 2]
    gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=axmat).attach_to(base)
    print(rm.unit_vector(np.array([0, 0, 0])))

    pos = np.array([.3, 0, 0])
    rotmat = rm.rotmat_from_euler(math.pi / 6, 0, 0)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    gen_frame_box([.1, .2, .3], homomat).attach_to(base)

    base.run()
