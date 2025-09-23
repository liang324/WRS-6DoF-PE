import copy
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath, BitMask32
from visualization.panda.world import ShowBase
import basis.robot_math as rm
import basis.data_adapter as da
import modeling.geometric_model as gm
import modeling.model_collection as mc
import modeling._panda_cdhelper as pcd
import modeling._ode_cdhelper as mcd


# 以下两个辅助函数无法正确查找碰撞位置,20211216
# TODO 检查它是否是由 mcd.update_pose中的错误 bullet 转换引起的
# import modeling._gimpact_cdhelper as mcd
# import modeling._bullet_cdhelper as mcd


class CollisionModel(gm.GeometricModel):
    """
    加载对象作为碰撞模型

    碰撞原语将自动生成
    注意:此类严重依赖于 Panda3D
          cdnp 碰撞检测原语的节点路径
          pdnp 网格和装饰的节点路径；装饰包括坐标框架、标记等
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

        :param initor: 初始化器,可以是碰撞模型或其他类型
        :param cdprimit_type: 碰撞原语类型,可选 'box', 'ball', 'cylinder', 'point_cloud', 'user_defined'
        :param cdmesh_type: 碰撞网格类型,可选 'aabb', 'obb', 'convex_hull', 'triangulation'
        :param expand_radius: 扩展半径,用于某些碰撞原语
        :param name: 模型名称
        :param userdefined_cdprimitive_fn: 用户定义的碰撞原语函数,如果 cdprimitive_type 为 'external',则使用此函数定义
                                           回调函数协议:返回 CollisionNode,可能包含多个 CollisionSolid
        :param btransparency: 是否启用透明度
        :param btwosided: 是否启用双面渲染

        date: 201290312, 20201212
        """
        if isinstance(initor, CollisionModel):
            # 如果初始化器是另一个碰撞模型,进行深拷贝
            self._name = copy.deepcopy(initor.name)
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._localframe = copy.deepcopy(initor.localframe)
            self._cdprimitive_type = copy.deepcopy(initor.cdprimitive_type)
            self._cdmesh_type = copy.deepcopy(initor.cdmesh_type)
        else:
            # 否则,使用父类的初始化方法
            super().__init__(initor=initor, name=name, btransparency=btransparency, btwosided=btwosided)
            self._cdprimitive_type, collision_node = self._update_cdprimit(cdprimit_type,
                                                                           expand_radius,
                                                                           userdefined_cdprimitive_fn)
            # 使用 pdnp.getChild 而不是新的 self._cdnp 变量,因为碰撞节点路径不兼容深拷贝
            self._objpdnp.attachNewNode(collision_node)
            self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))
            self.cdmesh_type = cdmesh_type
            self._localframe = None

    def _update_cdprimit(self, cdprimitive_type, expand_radius, userdefined_cdprimitive_fn):
        """
        更新碰撞原语

        :param cdprimitive_type: 碰撞原语类型,可选 'box', 'surface_balls', 'cylinder', 'polygons', 'point_cloud', 'user_defined'
        :param expand_radius: 扩展半径,用于某些碰撞原语
        :param userdefined_cdprimitive_fn: 用户定义的碰撞原语函数
        :return: 碰撞原语类型和碰撞节点
        """
        if cdprimitive_type is not None and cdprimitive_type not in ['box',
                                                                     'surface_balls',
                                                                     'cylinder',
                                                                     'polygons',
                                                                     'point_cloud',
                                                                     'user_defined']:
            raise ValueError("错误的碰撞模型原语类型名称！")
        if cdprimitive_type == 'surface_balls':
            if expand_radius is None:
                expand_radius = 0.015  # 默认扩展半径
            collision_node = pcd.gen_surfaceballs_cdnp(self.objtrm, name='cdnp_surface_ball', radius=expand_radius)
        else:
            if expand_radius is None:
                expand_radius = 0.002  # 默认扩展半径
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
        获取碰撞原语类型
        """
        return self._cdprimitive_type

    @property
    def cdmesh_type(self):
        """
        获取碰撞网格类型
        """
        return self._cdmesh_type

    @cdmesh_type.setter
    def cdmesh_type(self, cdmesh_type):
        """
        设置碰撞网格类型

        :param cdmesh_type: 碰撞网格类型,可选 'aabb', 'obb', 'convex_hull', 'triangles'
        """
        if cdmesh_type is not None and cdmesh_type not in ['aabb',
                                                           'obb',
                                                           'convex_hull',
                                                           'triangles']:
            raise ValueError("错误的网格碰撞模型类型名称！")
        self._cdmesh_type = cdmesh_type

    @property
    def cdnp(self):
        """
        获取碰撞检测节点路径
        """
        return self._objpdnp.getChild(1)  # child-0 = pdnp_raw, child-1 = cdnp

    @property
    def cdmesh(self):
        """
        获取碰撞网格
        """
        return mcd.gen_cdmesh_vvnf(*self.extract_rotated_vvnf())

    def extract_rotated_vvnf(self):
        """
        提取旋转后的顶点、顶点法线和面

        根据当前的碰撞网格类型(`cdmesh_type`),选择不同的几何体进行处理
        :return: 旋转后的顶点、顶点法线和面
        """
        if self.cdmesh_type == 'aabb':
            objtrm = self.objtrm.bounding_box
        elif self.cdmesh_type == 'obb':
            objtrm = self.objtrm.bounding_box_oriented
        elif self.cdmesh_type == 'convex_hull':
            objtrm = self.objtrm.convex_hull
        elif self.cdmesh_type == 'triangles':
            objtrm = self.objtrm
        homomat = self.get_homomat()
        vertices = rm.homomat_transform_points(homomat, objtrm.vertices)
        vertex_normals = rm.homomat_transform_points(homomat, objtrm.vertex_normals)
        faces = objtrm.faces
        return vertices, vertex_normals, faces

    def change_cdprimitive_type(self, cdprimitive_type='ball', expand_radius=.01, userdefined_cdprimitive_fn=None):
        """
        更改碰撞原语类型

        :param cdprimitive_type: 碰撞原语类型
        :param expand_radius: 扩展半径
        :param userdefined_cdprimitive_fn: 用户定义的碰撞原语函数,仅在 cdprimitive_type 为 'userdefined' 时使用
        :return: 无返回值

        author: weiwei
        date: 20210116
        """
        self._cdprimitive_type, cdnd = self._update_cdprimit(cdprimitive_type, expand_radius,
                                                             userdefined_cdprimitive_fn)
        # 使用 _objpdnp.getChild 而不是新的 self._cdnp 变量,因为碰撞节点路径与 deepcopy 不兼容
        self.cdnp.removeNode()
        self._objpdnp.attachNewNode(cdnd)
        self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))

    def change_cdmesh_type(self, cdmesh_type='convex_hull'):
        """
        更改碰撞网格类型

        :param cdmesh_type: 碰撞网格类型
        :return: 无返回值

        author: weiwei
        date: 20210117
        """
        self.cdmesh_type = cdmesh_type

    def copy_cdnp_to(self, nodepath, homomat=None, clearmask=False):
        """
        将碰撞检测节点路径复制到指定的节点路径

        :param nodepath: 父节点路径
        :param homomat: 允许指定一个特殊的齐次矩阵,以虚拟表示与网格不同的姿态
        :param clearmask: 是否清除碰撞掩码
        :return: 返回附加了碰撞节点的节点路径

        author: weiwei
        date: 20180811
        """
        returnnp = nodepath.attachNewNode(copy.deepcopy(self.cdnp.getNode(0)))
        if clearmask:
            returnnp.node().setCollideMask(0x00)
        else:
            returnnp.node().setCollideMask(self.cdnp.getCollideMask())
        if homomat is None:
            returnnp.setMat(self._objpdnp.getMat())
        else:
            returnnp.setMat(da.npmat4_to_pdmat4(homomat))  # 设置给定齐次矩阵后,比例重置为 1 1 1
            returnnp.setScale(self._objpdnp.getScale())
        return returnnp

    def is_pcdwith(self, objcm):
        """
        判断此碰撞模型的原语是否与给定碰撞模型的原语发生碰撞

        :param objcm: 一个或多个碰撞模型对象
        :return: 是否发生碰撞

        author: weiwei
        date: 20201116
        """
        return pcd.is_collided(self, objcm)

    def attach_to(self, obj):
        """
        将碰撞模型附加到指定对象

        :param obj: 可以是 ShowBase 实例或 ModelCollection 实例
        """
        if isinstance(obj, ShowBase):
            # 如果是 ShowBase 实例,则附加到渲染节点
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, mc.ModelCollection):
            # 如果是 ModelCollection 实例,则添加到模型集合中
            obj.add_cm(self)
        else:
            print(
                "必须是 ShowBase、modeling.StaticGeometricModel、GeometricModel、CollisionModel 或 CollisionModelCollection 的实例！")

    def detach(self):
        """
        从当前父节点分离碰撞模型
        """
        # TODO: 是否需要从模型集合中分离？
        self._objpdnp.detachNode()

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
        判断此碰撞模型的网格是否与给定碰撞模型的网格发生碰撞

        :param objcm_list: 一个或多个碰撞模型对象
        :param toggle_contacts: 如果为 True,则返回接触点列表
        :return: 是否发生碰撞,以及(可选的)接触点

        author: weiwei
        date: 20201116
        """
        if not isinstance(objcm_list, list):
            objcm_list = [objcm_list]
        for objcm in objcm_list:
            iscollided, contact_points = mcd.is_collided(self.cdmesh, objcm.cdmesh)
            if iscollided and toggle_contacts:
                return [True, contact_points]
            elif iscollided:
                return True
        return [False, []] if toggle_contacts else False

    def ray_hit(self, point_from, point_to, option="all"):
        """
        检查从 point_from 到 point_to 的线段与网格的交点

        :param point_from: 起始点,1x3 的 numpy 数组
        :param point_to: 结束点
        :param option: "all" 或 "closest",指定返回所有交点或最近的交点
        :return: 交点和法线

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
        显示碰撞网格的线框模型
        """
        vertices, vertex_normals, faces = self.extract_rotated_vvnf()
        objwm = gm.WireFrameModel(da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
        self._tmp_shown_cdmesh = objwm.attach_to(base)

    def unshow_cdmesh(self):
        """
        隐藏碰撞网格的线框模型
        """
        if hasattr(self, '_tmp_shown_cdmesh'):
            self._tmp_shown_cdmesh.detach()

    def is_mboxcdwith(self, objcm):
        """
        检查此模型的包围盒是否与给定模型的包围盒发生碰撞
        """
        raise NotImplementedError

    def copy(self):
        """
        创建此碰撞模型的深拷贝
        """
        return copy.deepcopy(self)


def gen_box(extent=np.array([.1, .1, .1]), homomat=np.eye(4), rgba=np.array([1, 0, 0, 1])):
    """
    生成一个立方体碰撞模型

    :param extent: 立方体的尺寸,默认为 [0.1, 0.1, 0.1]
    :param homomat: 变换矩阵,默认为单位矩阵
    :param rgba: 颜色和透明度,默认为红色 [1, 0, 0, 1]
    :return: 立方体的碰撞模型

    author: weiwei
    date: 20201202
    """
    box_sgm = gm.gen_box(extent=extent, homomat=homomat, rgba=rgba)
    box_cm = CollisionModel(box_sgm)
    return box_cm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1]):
    """
    生成一个球体碰撞模型

    :param pos: 球体的中心位置,默认为 [0, 0, 0]
    :param radius: 球体的半径,默认为 0.01
    :param rgba: 颜色和透明度,默认为红色 [1, 0, 0, 1]
    :return: 球体的碰撞模型

    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_sgm = gm.gen_sphere(pos=pos, radius=radius, rgba=rgba)
    sphere_cm = CollisionModel(sphere_sgm)
    return sphere_cm


def gen_stick(spos=np.array([.0, .0, .0]),
              epos=np.array([.0, .0, .1]),
              thickness=.005, type="rect",
              rgba=[1, 0, 0, 1],
              sections=8):
    """
    生成一个棍状碰撞模型

    :param spos: 棍子的起始位置,默认为 [0, 0, 0]
    :param epos: 棍子的结束位置,默认为 [0, 0, 0.1]
    :param thickness: 棍子的厚度,默认为 0.005
    :param type: 棍子的类型,默认为 "rect"
    :param rgba: 颜色和透明度,默认为红色 [1, 0, 0, 1]
    :param sections: 棍子的截面数,默认为 8
    :return: 棍状的碰撞模型

    author: weiwei
    date: 20210328
    """
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
    bunnycm.set_rgba([1, 0, 0.0, 1])
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)


    bunnycm1 = CollisionModel(objpath, cdprimit_type="cylinder")
    bunnycm1.set_rgba([0, 1, 0, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycm2 = bunnycm1.copy()
    bunnycm2.change_cdprimitive_type(cdprimitive_type='surface_balls')
    bunnycm2.set_rgba([0, 0.7, 0.7, 1.0])
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnycm2.set_pos(np.array([0, .2, 0]))
    bunnycm2.set_rotmat(rotmat)

    bunnycm.attach_to(base)
    bunnycm1.attach_to(base)
    bunnycm2.attach_to(base)
    # bunnycm.show_cdprimit()
    # bunnycm1.show_cdprimit()
    # bunnycm2.show_cdprimit()

    bunnycmpoints, _ = bunnycm.sample_surface()
    bunnycm1points, _ = bunnycm1.sample_surface()
    bunnycm2points, _ = bunnycm2.sample_surface()
    bpcm = gm.GeometricModel(bunnycmpoints)
    bpcm1 = gm.GeometricModel(bunnycm1points)
    bpcm2 = gm.GeometricModel(bunnycm2points)
    bpcm.attach_to(base)
    bpcm1.attach_to(base)
    bpcm2.attach_to(base)

    # bunnycm2.show_cdmesh(type='box')
    # bunnycm.show_cdmesh(type='box')
    # bunnycm1.show_cdmesh(type='convexhull')
    tic = time.time()
    result = bunnycm2.is_mcdwith([bunnycm, bunnycm1], toggle_contacts=True)
    print("碰撞检测的结果:", result)

    toc = time.time()
    print("meshes cd cost: ", toc - tic)
    tic = time.time()
    bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    toc = time.time()
    print("primitive cd cost: ", toc - tic)

    gen_box().attach_to(base)
    base.run()
