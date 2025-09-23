import copy
import numpy as np
import numpy.typing as npt
from panda3d.bullet import BulletTriangleMeshShape, BulletRigidBodyNode, BulletTriangleMesh
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath, BitMask32
from visualization.panda.world import ShowBase
import basis.robot_math as rm
import basis.data_adapter as da
import modeling.geometric_model as gm
import modeling.model_collection as mc
import modeling._panda_cdhelper as pcd
import modeling._ode_cdhelper as mcd
import warnings as wrn


# 以下两个助手无法正确找到碰撞位置,20211216
# TODO 检查是否由 mcd.update_pose 中的错误子弹变换导致
# import modeling._gimpact_cdhelper as mcd
# import modeling._bullet_cdhelper as mcd

class CollisionModel(gm.GeometricModel):
    """
    将对象作为碰撞模型加载,两个碰撞原语将自动生成

    注意:此类严重依赖于 Panda3D
    cdnp 碰撞检测原语的节点路径
    pdnp 网格+装饰的节点路径；装饰 = 坐标框架、标记等
    pdnp 网格的节点路径

    author: weiwei
    date: 20190312
    """

    def __init__(self,
                 initor,
                 cdprimit_type='box',
                 cdmesh_type='triangles',
                 expand_radius=None,
                 name="auto",
                 userdefined_cdprimitive_fn=None,
                 btransparency=True,
                 btwosided=False):
        """
        初始化碰撞模型

        :param initor: 初始化器,可以是路径或其他 CollisionModel 对象
        :param cdprimit_type: 碰撞原语类型,选项有 'box', 'ball', 'cylinder', 'point_cloud', 'user_defined'
        :param cdmesh_type: 碰撞网格类型,选项有 'aabb', 'obb', 'convex_hull', 'triangles'
        :param expand_radius: 扩展半径
        :param name: 名称
        :param userdefined_cdprimitive_fn: 用户自定义的碰撞原语函数,如果 cdprimitive_type = external,则在提供的函数中定义碰撞体
                                           回调函数协议:返回 CollisionNode,可能有多个 CollisionSolid
        :param btransparency: 是否透明
        :param btwosided: 是否双面

        date: 201290312, 20201212
        """
        if isinstance(initor, CollisionModel):
            self._name = copy.deepcopy(initor.name)
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._localframe = copy.deepcopy(initor.localframe)
            self._cdprimitive_type = copy.deepcopy(initor.cdprimitive_type)
            self._cdmesh_type = copy.deepcopy(initor.cdmesh_type)

            # TODO 需要考虑同时设置init和types的TODO异常
            # 此外,未使用 copy.deepcopy,因为它会调用已弃用的 getdata 方法,
            # 详情请参阅我在 https://discourse.panda3d.org/t/ode-odetrimeshdata-problem/28232 上的问题和评论
            # weiwei, 20220105

            self._cdmesh = mcd.copy_cdmesh(initor.cdmesh)
        else:
            super().__init__(initor=initor, name=name, btransparency=btransparency, btwosided=btwosided)
            self._cdprimitive_type, collision_node = self._update_cdprimit(cdprimit_type,
                                                                           expand_radius,
                                                                           userdefined_cdprimitive_fn)
            # 使用 pdnp.getChild 而不是新的 self._cdnp 变量,因为冲突 nodepath 与 deepcopy 不兼容

            self._objpdnp.attachNewNode(collision_node)
            self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))
            self._cdmesh_type = cdmesh_type
            self._cdmesh = mcd.gen_cdmesh_vvnf(*self.extract_rotated_vvnf())
            self._localframe = None
            # 重新初始化 self._cdmesh,同时忽略初始化器类型,重新初始化有助于避免由深度复制引起的烦人的 ode 警告.

    def _update_cdprimit(self, cdprimitive_type, expand_radius, userdefined_cdprimitive_fn):
        """
        更新碰撞原语

        :param cdprimitive_type: 碰撞原语的类型,可以是 'box', 'surface_balls', 'cylinder', 'polygons', 'point_cloud', 'user_defined'
        :param expand_radius: 扩展半径,用于调整碰撞原语的大小
        :param userdefined_cdprimitive_fn: 用户自定义的碰撞原语生成函数
        :return: 碰撞原语的类型和生成的碰撞节点
        要生成的碰撞原语类型,如 ‘box（盒子）’、‘surface_balls（表面分布球）’、
        ‘cylinder（圆柱）’、‘polygons（多边形面片）’、‘point_cloud（点云）’、以及 ‘user_defined（用户自定义）’.

        """
        # 检查碰撞原语类型是否合法
        if cdprimitive_type is not None and cdprimitive_type not in ['box',
                                                                     'surface_balls',
                                                                     'cylinder',
                                                                     'polygons',
                                                                     'point_cloud',
                                                                     'user_defined']:
            raise ValueError("错误的碰撞模型类型名称!")

        # 根据类型生成相应的碰撞节点
        if cdprimitive_type == 'surface_balls':
            if expand_radius is None:
                expand_radius = 0.015
            collision_node = pcd.gen_surfaceballs_cdnp(self.objtrm, name='cdnp_surface_ball', radius=expand_radius)
        else:
            if expand_radius is None:
                expand_radius = 0.002
            if cdprimitive_type == "box":
                collision_node = pcd.gen_box_cdnp(self.objpdnp_raw, name='cdnp_box', radius=expand_radius)
            if cdprimitive_type == "cylinder":
                collision_node = pcd.gen_cylindrical_cdnp(self.objpdnp_raw, name='cdnp_cyl', radius=expand_radius)
            if cdprimitive_type == "polygons":
                collision_node = pcd.gen_polygons_cdnp(self.objpdnp_raw, name='cdnp_plys', radius=expand_radius)
            if cdprimitive_type == "point_cloud":
                collision_node = pcd.gen_pointcloud_cdnp(self.objtrm, name='cdnp_ptc', radius=expand_radius)
            if cdprimitive_type == "user_defined":
                collision_node = userdefined_cdprimitive_fn(name="cdnp_usrdef", radius=expand_radius)
        return cdprimitive_type, collision_node

    @property
    def cdprimitive_type(self):
        """
        获取碰撞原语的类型

        :return: 碰撞原语的类型
        """
        return self._cdprimitive_type

    @property
    def cdmesh_type(self):
        """
        获取碰撞网格的类型

        :return: 碰撞网格的类型
        """
        return self._cdmesh_type

    @cdmesh_type.setter
    def cdmesh_type(self, cdmesh_type):
        """
        设置碰撞网格的类型

        :param cdmesh_type: 碰撞网格的类型,可以是 'aabb', 'obb', 'convex_hull', 'triangles'
        """
        if cdmesh_type is not None and cdmesh_type not in ['aabb',
                                                           'obb',
                                                           'convex_hull',
                                                           'triangles']:
            raise ValueError("错误的网格碰撞模型类型名称！")
        self._cdmesh_type = cdmesh_type
        self._cdmesh = mcd.gen_cdmesh_vvnf(*self.extract_rotated_vvnf())

    @property
    def cdnp(self):
        """
        获取碰撞节点路径

        :return: 碰撞节点路径的子节点
        """
        return self._objpdnp.getChild(1)  # child-0 = pdnp_raw, child-1 = cdnp

    @property
    def cdmesh(self):
        """
        使用 ODE 获取碰撞网格

        :return: 碰撞网格

        author: weiwei
        date: 20211215
        """
        return self._cdmesh

    def set_scale(self, scale=[1, 1, 1]):
        """
        设置对象的缩放比例

        :param scale: 缩放比例,默认为 [1, 1, 1]
        """
        self._objpdnp.setScale(scale[0], scale[1], scale[2])
        self._objtrm.apply_scale(scale)
        # 更新碰撞网格以反映新的缩放
        self._cdmesh = mcd.gen_cdmesh_vvnf(*self.extract_rotated_vvnf())

    def get_scale(self):
        """
        获取对象的缩放比例

        :return: 缩放比例的 NumPy 数组
        """
        return da.pdv3_to_npv3(self._objpdnp.getScale())

    def set_pos(self, pos: npt.NDArray = np.zeros(3)):
        """
        设置对象的位置

        :param pos: 位置的 NumPy 数组,默认为 [0, 0, 0]
        """
        self._objpdnp.setPos(pos[0], pos[1], pos[2])
        # 更新碰撞网格的位置
        mcd.update_pose(self._cdmesh, self._objpdnp)

    def set_rotmat(self, rotmat: npt.NDArray = np.eye(3)):
        """
        设置对象的旋转矩阵

        :param rotmat: 旋转矩阵的 NumPy 数组,默认为单位矩阵
        """
        self._objpdnp.setQuat(da.npmat3_to_pdquat(rotmat))
        # 更新碰撞网格的姿态
        mcd.update_pose(self._cdmesh, self._objpdnp)

    def set_pose(self, pos: npt.NDArray = np.zeros(3), rotmat: npt.NDArray = np.eye(3)):
        """
        设置对象的位置和旋转矩阵

        :param pos: 位置的 NumPy 数组,默认为 [0, 0, 0]
        :param rotmat: 旋转矩阵的 NumPy 数组,默认为单位矩阵
        """
        self._objpdnp.setPosQuat(da.npv3_to_pdv3(pos), da.npmat3_to_pdquat(rotmat))
        # 更新碰撞网格的姿态
        mcd.update_pose(self._cdmesh, self._objpdnp)

    def set_homomat(self, npmat4):
        """
        设置对象的齐次变换矩阵

        :param npmat4: 齐次变换矩阵的 NumPy 数组
        """
        self._objpdnp.setPosQuat(da.npv3_to_pdv3(npmat4[:3, 3]), da.npmat3_to_pdquat(npmat4[:3, :3]))
        mcd.update_pose(self._cdmesh, self._objpdnp)

    def set_rpy(self, roll, pitch, yaw):
        """
        使用 RPY(滚转、俯仰、偏航)设置对象的姿态

        :param roll: 滚转角度(弧度)
        :param pitch: 俯仰角度(弧度)
        :param yaw: 偏航角度(弧度)

        author: weiwei
        date: 20190513
        """
        npmat3 = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self.set_rotmat(npmat3)
        mcd.update_pose(self._cdmesh, self._objpdnp)

    def extract_rotated_vvnf(self, cdmesh_type=None):
        """
        允许提取指定的cdmesh_type或self.cdmesh_type值之后的VVNF

        :param cdmesh_type: 碰撞网格类型,可选
        :return: 顶点、顶点法线和面的元组

        author: weiwei
        date: 20211215
        """
        if cdmesh_type is None:
            cdmesh_type = self.cdmesh_type
        if cdmesh_type == 'aabb':
            objtrm = self.objtrm.bounding_box
        elif cdmesh_type == 'obb':
            objtrm = self.objtrm.bounding_box_oriented
        elif cdmesh_type == 'convex_hull':
            objtrm = self.objtrm.convex_hull
        elif cdmesh_type == 'triangles':
            objtrm = self.objtrm
        homomat = self.get_homomat()
        # 应用齐次变换到顶点和顶点法线
        vertices = rm.homomat_transform_points(homomat, objtrm.vertices)
        vertex_normals = rm.homomat_transform_points(homomat, objtrm.vertex_normals)
        faces = objtrm.faces
        return vertices, vertex_normals, faces

    def change_cdprimitive_type(self, cdprimitive_type='ball', expand_radius=.01, userdefined_cdprimitive_fn=None):
        """
        更改碰撞原语的类型

        :param cdprimitive_type: 碰撞原语的类型,默认为 'ball'
        :param expand_radius: 扩展半径,用于调整碰撞原语的大小
        :param userdefined_cdprimitive_fn: 用户定义的碰撞原语函数,仅在 cdprimitive_type 为 'userdefined' 时使用

        :return:
        author: weiwei
        date: 20210116
        """
        # 更新碰撞原语并获取新的碰撞节点
        self._cdprimitive_type, cdnd = self._update_cdprimit(cdprimitive_type, expand_radius,
                                                             userdefined_cdprimitive_fn)
        # 使用 _objpdnp.getChild 而不是新的 self._cdnp 变量,因为碰撞节点路径与 deepcopy 不兼容
        self.cdnp.removeNode()
        self._objpdnp.attachNewNode(cdnd)
        self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))

    def change_cdmesh_type(self, cdmesh_type='convex_hull'):
        """
        更改碰撞网格的类型

        :param cdmesh_type: 碰撞网格的类型,默认为 'convex_hull'

        author: weiwei
        date: 20210117
        """
        self.cdmesh_type = cdmesh_type

    def copy_cdnp_to(self, nodepath, homomat=None, clearmask=False):
        """
        将碰撞节点复制到指定的节点路径

        :param nodepath: 父节点路径
        :param homomat: 允许指定一个特殊的齐次变换矩阵,以虚拟地表示与网格不同的姿态
        :param clearmask: 是否清除碰撞掩码
        :return: 返回附加到给定节点路径的节点路径

        author: weiwei
        date: 20180811
        """
        # 复制碰撞节点并附加到指定的节点路径
        returnnp = nodepath.attachNewNode(copy.deepcopy(self.cdnp.getNode(0)))
        if clearmask:
            returnnp.node().setCollideMask(0x00)
        else:
            returnnp.node().setCollideMask(self.cdnp.getCollideMask())
        if homomat is None:
            returnnp.setMat(self._objpdnp.getMat())
        else:
            returnnp.setMat(da.npmat4_to_pdmat4(homomat))  # 设置齐次变换矩阵后缩放重置为 1 1 1
            returnnp.setScale(self._objpdnp.getScale())
        return returnnp

    def is_pcdwith(self, objcm):
        """
        检查此碰撞模型的原语是否与给定碰撞模型的原语发生碰撞

        :param objcm: 一个或多个碰撞模型对象
        :return: 返回碰撞检测结果

        author: weiwei
        date: 20201116
        """
        return pcd.is_collided(self, objcm)

    def attach_to(self, obj):
        """
        将当前对象附加到指定的对象上

        :param obj: 可以是 ShowBase 实例或 ModelCollection 实例
        """
        if isinstance(obj, ShowBase):
            # 如果是 ShowBase 实例,将对象附加到渲染树中
            # for rendering to base.render
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, mc.ModelCollection):
            # 如果是 ModelCollection 实例,将当前碰撞模型添加到集合中
            obj.add_cm(self)
        else:
            print(
                "必须是 ShowBase、modeling.StaticGeometricModel、GeometricModel、CollisionModel 或 CollisionModelCollection!")

    def detach(self):
        """
        从当前附加的节点中分离对象
        """
        self._objpdnp.detachNode()

    def remove(self):
        """
        移除碰撞模型的节点
        """
        self._objpdnp.removeNode()



    def show_cdprimit(self):
        """
        显示碰撞节点
        """
        self.cdnp.show()

    def unshow_cdprimit(self):
        """
        隐藏碰撞节点
        """
        self.cdnp.hide()

    def is_mcdwith(self, objcm_list, toggle_contacts=False):
        """
        检测当前碰撞模型与给定的一个或多个碰撞模型之间是否发生碰撞

        :param objcm_list: 一个或多个碰撞模型对象,可以是单个对象或对象列表
        :param toggle_contacts: 布尔值,指示是否返回接触点列表
        :return: 如果 `toggle_contacts` 为 True,返回 [碰撞结果, 接触点列表]；否则仅返回碰撞结果

        author: weiwei
        date: 20201116
        """
        # 如果 objcm_list 不是列表,将其转换为列表
        if not isinstance(objcm_list, list):
            objcm_list = [objcm_list]

        # 遍历每个目标碰撞模型对象
        for objcm in objcm_list:
            # 检测当前模型与目标模型之间的碰撞
            iscollided, contact_points = mcd.is_collided(self.cdmesh, objcm.cdmesh)

            # 如果发生碰撞并且需要接触点信息
            if iscollided and toggle_contacts:
                return [True, contact_points]
            # 如果发生碰撞但不需要接触点信息
            elif iscollided:
                return True
        return [False, []] if toggle_contacts else False

    def ray_hit(self, point_from, point_to, option="all"):
        """
        检查从 point_from 到 point_to 的线段与网格之间的交点

        :param point_from: 起始点,1x3 的 numpy 数组
        :param point_to: 终止点
        :param option: "all" 或 "closest",指示返回所有交点或最近的交点
        :return: 交点和法线信息

        author: weiwei
        date: 20210504
        """
        if option == "all":
            contact_points, contact_normals = mcd.rayhit_all(point_from, point_to, self)
            return contact_points, contact_normals
        elif option == "closest":
            contact_point, contact_normal = mcd.rayhit_closet(point_from, point_to, self)
            return contact_point, contact_normal

    def show_cdmesh(self):
        """
        显示碰撞网格
        """
        vertices, vertex_normals, faces = self.extract_rotated_vvnf()
        objwm = gm.WireFrameModel(da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
        self._tmp_shown_cdmesh = objwm.attach_to(base)

    def unshow_cdmesh(self):
        """
        隐藏碰撞网格
        """
        if hasattr(self, '_tmp_shown_cdmesh'):
            self._tmp_shown_cdmesh.detach()

    def is_mboxcdwith(self, objcm):
        """
        检测当前模型是否与给定模型的包围盒发生碰撞
        """
        raise NotImplementedError

    def copy(self):
        """
        创建当前碰撞模型的副本

        :return: 新的碰撞模型对象
        """
        return CollisionModel(self)

    def get_com(self):
        """
        获取碰撞模型的质心(中心质量)

        :return: 质心的坐标
        """
        com = self._objtrm.center_mass
        return com

    def __deepcopy__(self, memodict={}):
        """
        重写 __deepcopy__ 方法以绕过 ode 废弃函数的问题

        :param memodict: 深拷贝时使用的字典
        :return: 当前对象的副本

        author: weiwei
        date: 20220115toyonaka
        """
        return self.copy()


def gen_box(extent=np.array([.1, .1, .1]), homomat=np.eye(4), rgba=np.array([1, 0, 0, 1])):
    """
    生成一个包围盒碰撞模型

    :param extent: 包围盒的尺寸,numpy 数组 [宽度, 高度, 深度]
    :param homomat: 变换矩阵,默认是单位矩阵
    :param rgba: 颜色,numpy 数组 [红, 绿, 蓝, 透明度]
    :return: 包围盒碰撞模型对象

    author: weiwei
    date: 20201202
    """
    box_sgm = gm.gen_box(extent=extent, homomat=homomat, rgba=rgba)
    box_cm = CollisionModel(box_sgm)
    return box_cm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1]):
    """
    生成一个球形碰撞模型

    :param pos: 球心的位置,numpy 数组 [x, y, z]
    :param radius: 球的半径
    :param rgba: 颜色,列表 [红, 绿, 蓝, 透明度]
    :return: 球形碰撞模型对象

    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_sgm = gm.gen_sphere(pos=pos, radius=radius, rgba=rgba)
    sphere_cm = CollisionModel(sphere_sgm)
    return sphere_cm


def gen_stick(spos=np.array([.0, .0, .0]),
              epos=np.array([.0, .0, .1]),
              thickness=.005, type="round",
              rgba=[1, 0, 0, 1],
              sections=8):
    """
    生成一个棍状的碰撞模型

    :param spos: 起始点坐标,numpy 数组 [x, y, z]
    :param epos: 终止点坐标,numpy 数组 [x, y, z]
    :param thickness: 棍子的厚度
    :param type: 棍子的类型,默认为 "round"
    :param rgba: 颜色,列表 [红, 绿, 蓝, 透明度]
    :param sections: 棍子的截面数,用于控制圆形截面的细分程度
    :return: 棍状碰撞模型对象

    author: weiwei
    date: 20210328
    """
    # 使用几何模块生成棍子的几何形状
    stick_sgm = gm.gen_stick(spos=spos, epos=epos, thickness=thickness, type=type, rgba=rgba, sections=sections)
    stick_cm = CollisionModel(stick_sgm)
    return stick_cm





if __name__ == "__main__":
    import os
    import math
    import time
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0], toggle_debug=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnycm = CollisionModel(objpath, cdprimit_type='polygons')
    bunnycm.set_rgba([0.7, 0, 0.0, .2])
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)
    bunnycm.show_cdprimit()
    bunnycm.attach_to(base)
    # base.run()

    bunnycm1 = CollisionModel(objpath, cdprimit_type="cylinder")
    bunnycm1.set_rgba([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, 0, 0]))
    bunnycm1.set_rotmat(rotmat)
    bunnycm1.attach_to(base)

    bunnycm2 = bunnycm1.copy()
    bunnycm2.change_cdprimitive_type(cdprimitive_type='surface_balls')
    bunnycm2.set_rgba([0, 0.7, 0.7, 1.0])
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnycm2.set_pos(np.array([0, .0, 0]))
    bunnycm2.set_rotmat(rotmat)
    bunnycm2.attach_to(base)

    bunnycm.show_cdprimit()
    bunnycm1.show_cdprimit()
    bunnycm2.show_cdprimit()
    base.run()

    bunnycmpoints, _ = bunnycm.sample_surface()
    bunnycm1points, _ = bunnycm1.sample_surface()
    bunnycm2points, _ = bunnycm2.sample_surface()

    bpcm = gm.GeometricModel(bunnycmpoints)
    bpcm1 = gm.GeometricModel(bunnycm1points)
    bpcm2 = gm.GeometricModel(bunnycm2points)

    bpcm.attach_to(base)
    bpcm1.attach_to(base)
    bpcm2.attach_to(base)
    base.run()

    # bunnycm2.show_cdmesh(type='box')
    # bunnycm.show_cdmesh(type='box')
    # bunnycm1.show_cdmesh(type='convexhull')
    # tic = time.time()
    # bunnycm2.is_mcdwith([bunnycm, bunnycm1])
    # toc = time.time()
    # print("meshes cd cost: ", toc - tic)
    # tic = time.time()
    # bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    # toc = time.time()
    # print("primitive cd cost: ", toc - tic)

    # gen_box().attach_to(base)
    base.run()
