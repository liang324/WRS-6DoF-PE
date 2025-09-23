import copy
import math
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.bullet import BulletWorld
import numpy as np
import modeling.geometric_model as gm
import modeling.dynamics.bullet.bdbody as bdb
from visualization.panda.world import ShowBase
import basis.data_adapter as da


class BDModel(object):
    """
    将一个对象加载为 Bullet 动力学模型

    author: weiwei
    date: 20190627
    """

    def __init__(self,
                 objinit,
                 mass=None,
                 restitution=0,
                 allowdeactivation=False,
                 allowccd=True,
                 friction=.2,
                 dynamic=True,
                 physics_scale: None or float = 1e3,
                 type="convex",
                 name="bdm"):
        """
        初始化 BDModel

        :param objinit: GeometricModel(CollisionModel 也适用)
        :param mass: 质量
        :param restitution: 恢复系数
        :param allowdeactivation: 是否允许停用
        :param allowccd: 是否允许连续碰撞检测
        :param friction: 摩擦系数
        :param dynamic: 是否为动态对象
        :param type: 碰撞检测类型,例如 "convex", "triangle", "box"
        :param name: 模型名称
        """
        if physics_scale is None:
            self.physics_scale = base.physics_scale
        else:
            self.physics_scale = physics_scale

        if isinstance(objinit, BDModel):
            # 如果是 BDModel 的实例,进行深拷贝
            self._gm = copy.deepcopy(objinit.gm)
            self._bdb = objinit.bdb.copy()
        elif isinstance(objinit, gm.GeometricModel):
            # 如果是 GeometricModel 的实例
            if mass is None:
                mass = 0
            self._gm = objinit
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allow_deactivation=allowdeactivation,
                                   allow_ccd=allowccd, friction=friction, dynamic=dynamic, name=name)
        else:
            # 其他情况,创建新的 GeometricModel
            if mass is None:
                mass = 0
            self._gm = gm.GeometricModel(objinit)
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allow_deactivation=allowdeactivation,
                                   allow_ccd=allowccd, friction=friction, dynamic=dynamic, name=name,
                                   physics_scale=self.physics_scale)

    @property
    def gm(self):
        """
        获取几何模型,只读属性

        :return: 几何模型对象
        """
        return self._gm

    @property
    def bdb(self):
        """
        获取 Bullet 动力学体,只读属性

        :return: BDBody 对象
        """
        return self._bdb

    @property
    def is_kinemetic(self):
        """
        检查刚体是否为运动学对象

        :return: 布尔值,表示刚体是否为运动学对象
        """
        return self._bdb.is_kinematic()

    def set_rgba(self, rgba):
        """
        设置对象的颜色

        :param rgba: 颜色值,格式为 (r, g, b, a)
        :return: None
        """
        self._gm.set_rgba(rgba)

    def clear_rgba(self):
        """
        清除对象的颜色设置

        :return: None
        """
        self._gm.clear_rgba()

    def get_rgba(self):
        """
        获取对象的颜色

        :return: 颜色值,格式为 (r, g, b, a)
        """
        return self._gm.get_rgba()

    def set_pos(self, npvec3):
        """
        设置对象的位置

        :param npvec3: 位置向量,格式为 numpy 数组 [x, y, z]
        :return: None
        """
        homomat_bdb = self._bdb.get_homomat()
        homomat_bdb[:3, 3] = npvec3
        self._bdb.set_homomat(homomat_bdb)
        self._gm.set_homomat(homomat_bdb)

    def get_pos(self):
        """
        获取对象的位置

        :return: 位置向量,格式为 numpy 数组 [x, y, z]
        """
        return self._bdb.get_pos()

    def set_homomat(self, npmat4):
        """
        设置对象的齐次变换矩阵

        :param npmat4: 齐次变换矩阵,格式为 4x4 numpy 数组
        :return: None
        """
        self._bdb.set_homomat(npmat4)
        self._gm.set_homomat(npmat4)

    def get_homomat(self):
        """
        获取对象的齐次变换矩阵

        :return: 齐次变换矩阵,格式为 4x4 numpy 数组
        """
        return self._bdb.get_homomat()

    def set_mass(self, mass):
        """
        设置对象的质量

        :param mass: 质量值
        :return: None
        """
        self._bdb.set_mass(mass)

    def set_restitution(self, restitution: float):
        """
        设置弹性系数(反弹系数)

        :param restitution: 弹性系数,范围在 0 到 1 之间
        :raises AssertionError: 如果弹性系数不在 0 到 1 之间,则抛出异常
        """
        assert 0 <= restitution <= 1, "The coefficient of restitution must be between 0 and 1"
        self._bdb.set_restitution(restitution)

    def set_friction(self, friction: float):
        """
        设置摩擦系数

        :param friction: 摩擦系数,必须为非负值
        :raises AssertionError: 如果摩擦系数为负,则抛出异常
        """
        assert 0 <= friction, "The friction coefficient must be non-negative"
        self._bdb.set_friction(friction)

    def set_speed(self, linear_speed: np.ndarray = None, angular_speed: np.ndarray = None):
        """
        设置物体的速度

        :param linear_speed: 线速度,默认为 None
        :param angular_speed: 角速度,默认为 None
        """
        if linear_speed is not None:
            self.set_linear_vel(linear_speed)
        if angular_speed is not None:
            self.set_angular_vel(angular_speed)

    def set_linear_vel(self, vel: np.ndarray):
        """
        设置物体的线速度

        :param vel: 线速度向量
        """
        self._bdb.set_linear_velocity(da.npv3_to_pdv3(vel) * self.physics_scale)

    def set_angular_vel(self, vel: np.ndarray):
        """
        设置物体的角速度

        :param vel: 角速度向量
        """
        self._bdb.set_angular_velocity(da.npv3_to_pdv3(vel) * self.physics_scale)

    def apply_central_impulse(self, force: np.ndarray):
        """
        对物体施加中心冲量

        :param force: 冲量向量
        """
        self._bdb.apply_central_impulse(da.npv3_to_pdv3(force) * self.physics_scale)

    def set_kinematic(self, kinematic: bool):
        """
        设置物体为运动学对象

        :param kinematic: 布尔值,表示是否设置为运动学对象
        """
        self._bdb.set_kinematic(kinematic)

    def attach_to(self, obj):
        """
        将对象附加到指定节点,obj 必须是 base.render

        :param obj: 目标节点
        :return: None

        author: weiwei
        date: 20190627
        """
        if isinstance(obj, ShowBase):
            # 用于渲染到 base.render
            self._gm.set_homomat(self.bdb.get_homomat())  # 获取动态更新
            self._gm.attach_to(obj)
        else:
            raise ValueError("必须是 ShowBase 类型！")

    def remove(self):
        """
        移除对象

        :return: None
        """
        self._gm.remove()

    def detach(self):
        """
        分离对象

        :return: None
        """
        self._gm.detach()

    def attach_to_bw(self, bw: BulletWorld):
        """
        将刚体附加到物理世界中

        :param bw: BulletWorld 实例,表示物理世界
        :raises AssertionError: 如果输入不是 BulletWorld 对象,则抛出异常
        """
        assert isinstance(bw, BulletWorld), "输入必须是BulletWorld对象！"
        bw.attach(self._bdb)

    def detach_from_bw(self, bw: BulletWorld):
        """
        从物理世界中移除刚体

        :param bw: BulletWorld 实例,表示物理世界
        :raises AssertionError: 如果输入不是 BulletWorld 对象,则抛出异常
        """
        assert isinstance(bw, BulletWorld), "输入必须是BulletWorld对象！"
        bw.remove(self._bdb)

    def start_physics(self):
        """
        开始物理模拟

        :return: None
        """
        base.physicsworld.attach(self._bdb)

    def end_physics(self):
        """
        结束物理模拟

        :return: None
        """
        base.physicsworld.remove(self._bdb)

    def sync_to_physical(self):
        """
        将几何模型的位置和方向与物理引擎中的刚体同步

        这个方法更新几何模型的变换矩阵,使其与刚体的变换矩阵一致
        """
        self._gm.set_homomat(self.bdb.get_homomat())

    def show_loc_frame(self):
        """
        显示局部坐标系

        :return: None
        """
        self._gm.showlocalframe()

    def unshow_loc_frame(self):
        """
        隐藏局部坐标系

        :return: None
        """
        self._gm.unshowlocalframe()

    def copy(self):
        """
        复制当前对象

        :return: 新的 BDModel 对象
        """
        return BDModel(self)


if __name__ == "__main__":
    import os
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import random

    # base = wd.World(cam_pos=[1000, 300, 1000], lookat_pos=[0, 0, 0], toggle_debug=True)
    base = wd.World(cam_pos=[.3, .3, 1], lookat_pos=[0, 0, 0], toggle_debug=False)
    base.setFrameRateMeter(True)
    objpath = os.path.join(basis.__path__[0], "objects", "bunnysim.stl")
    # objpath = os.path.join(basis.__path__[0], "objects", "block.stl")
    bunnycm = BDModel(objpath, mass=1, type="box")

    objpath2 = os.path.join(basis.__path__[0], "objects", "bowlblock.stl")
    bunnycm2 = BDModel(objpath2, mass=0, type="triangles", dynamic=False)
    bunnycm2.set_rgba(np.array([0, 0.7, 0.7, 1.0]))
    bunnycm2.set_pos(np.array([0, 0, 0]))
    bunnycm2.start_physics()
    base.attach_internal_update_obj(bunnycm2)


    def update(bunnycm, task):
        if base.inputmgr.keymap['space'] is True:
            for i in range(1):
                bunnycm1 = bunnycm.copy()
                bunnycm1.set_mass(.1)
                rndcolor = np.random.rand(4)
                rndcolor[-1] = 1
                bunnycm1.set_rgba(rndcolor)
                rotmat = rm.rotmat_from_euler(0, 0, math.pi / 12)
                z = math.floor(i / 100)
                y = math.floor((i - z * 100) / 10)
                x = i - z * 100 - y * 10
                print(x, y, z, "\n")
                bunnycm1.set_homomat(
                    rm.homomat_from_posrot(np.array([x * 0.015 - 0.07, y * 0.015 - 0.07, 0.35 + z * 0.015]), rotmat))
                base.attach_internal_update_obj(bunnycm1)
                bunnycm1.start_physics()
        base.inputmgr.keymap['space'] = False
        return task.cont


    gm.gen_frame().attach_to(base)
    taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

    base.run()
