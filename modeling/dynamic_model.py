import copy
import math
from direct.task.TaskManagerGlobal import taskMgr
import modeling.geometric_model as gm
import modeling.dynamics.bullet.bdbody as bdb
from modeling.dynamics.bullet.bdmodel import BDModel


class DynamicModel(gm.GeometricModel):
    """
    将一个对象加载为 Bullet 动力学模型

    author: weiwei
    date: 20190627
    """

    def __init__(self, initor, mass=None, betransparency=True, cm_cdtype="box", cm_expradius=None,
                 restitution=0, allowdeactivation=False, allowccd=True, friction=.2, dynamic=False,
                 dyn_cdtype="convex", name="bdm"):
        """
        初始化动态模型

        :param initor: 初始化器,可以是一个几何模型或其他对象
        :param mass: 质量,如果为None,则默认为0
        :param betransparency: 是否透明
        :param cm_cdtype: 碰撞检测类型,例如"box"
        :param cm_expradius: 碰撞检测扩展半径
        :param restitution: 恢复系数
        :param allowdeactivation: 是否允许停用
        :param allowccd: 是否允许连续碰撞检测
        :param friction: 摩擦系数
        :param dynamic: 是否为动态对象
        :param dyn_cdtype: 动态碰撞检测类型,例如"convex"或"triangle"
        :param name: 模型名称
        """
        # if isinstance(initor, BDModel):
        #     super().__init__(initor.objcm, )
        #     self.__objcm = copy.deepcopy(initor.objcm)
        #     self.__objbdb = initor.objbdb.copy()
        #     base.physicsworld.attach(self.__objbdb)
        # else:
        super().__init__(initor.objcm, btransparency=betransparency, type=cm_cdtype, cm_expradius=None,
                         name="defaultname")
        if mass is None:
            mass = 0
        self._bdb = bdb.BDBody(self.objcm, type=dyn_cdtype, mass=mass, restitution=restitution,
                               allow_deactivation=allowdeactivation, allow_ccd=allowccd, friction=friction, name="bdm")
        base.physicsworld.attach(self.__objbdb)

    @property
    def bdb(self):
        # 只读属性,返回BDBody对象
        return self._bdb

    def setpos(self, npvec3):
        """
        重写父类的setpos方法以额外操作bdb

        :param npvec3: 新的位置向量
        :return:
        """
        homomat_bdb = self._bdb.get_homomat()
        homomat_bdb[:3, 3] = npvec3
        self._bdb.set_homomat(homomat_bdb)
        super().sethomomat(homomat_bdb)

    def getpos(self):
        """
        获取对象的位置

        :return: 位置向量
        """
        homomat_bdb = self._bdb.pos()
        self._bdb.set_homomat(homomat_bdb)
        super().sethomomat(homomat_bdb)
        return self.__objcm.objnp.getPos()

    def setMat(self, pandamat4):
        """
        设置对象的变换矩阵

        :param pandamat4: Panda3D的4x4矩阵
        :return: None
        """
        self.__objbdb.set_homomat(base.pg.mat4ToNp(pandamat4))
        self.__objcm.objnp.setMat(pandamat4)
        # self.__objcm.objnp.setMat(base.pg.np4ToMat4(self.objbdb.gethomomat()))

    def sethomomat(self, npmat4):
        """
        设置对象的齐次变换矩阵

        :param npmat4: numpy的4x4矩阵
        :return: None
        """
        self.__objbdb.set_homomat(npmat4)
        self.__objcm.set_homomat(npmat4)

    def setRPY(self, roll, pitch, yaw):
        """
        使用RPY角设置对象的姿态

        :param roll: 滚转角,单位为度
        :param pitch: 俯仰角,单位为度
        :param yaw: 偏航角,单位为度
        :return: None

        author: weiwei
        date: 20190513
        """
        currentmat = self.__objbdb.get_homomat()
        currentmatnp = base.pg.mat4ToNp(currentmat)
        newmatnp = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self.setMat(base.pg.npToMat4(newmatnp, currentmatnp[:, 3]))

    def getRPY(self):
        """
        获取对象的RPY角

        :return: [r, p, y] 以度为单位

        author: weiwei
        date: 20190513
        """
        currentmat = self.objcm.getMat()
        currentmatnp = base.pg.mat4ToNp(currentmat)
        rpy = rm.euler_from_matrix(currentmatnp[:3, :3], axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def getMat(self, rel=None):
        """
        获取对象的变换矩阵

        :param rel: 相对节点
        :return: Panda3D的4x4矩阵
        """
        return self.objcm.getMat(rel)

    def gethomomat(self, rel=None):
        """
        获取对象的齐次变换矩阵

        :param rel: 相对节点
        :return: numpy的4x4矩阵
        """
        pandamat4 = self.getMat(rel)
        return base.pg.mat4ToNp(pandamat4)

    def setMass(self, mass):
        """
        设置对象的质量

        :param mass: 质量值
        :return: None
        """
        self.__objbdb.setMass(mass)

    def reparentTo(self, objnp):
        """
        将对象重新挂载到指定节点,objnp必须是base.render

        :param objnp: 目标节点
        :return: None

        author: weiwei
        date: 20190627
        """
        # if isinstance(objnp, cm.CollisionModel):
        #     self.__objcm.objnp.reparentTo(objnp.objnp)
        # elif isinstance(objnp, NodePath):
        #     self.__objcm.objnp.reparentTo(objnp)
        # else:
        #     print("NodePath.reparent_to() argument 1 must be environment.CollisionModel or panda3d.core.NodePath")
        if objnp is not base.render:
            print("此Bullet动力学模型不支持挂载到非base.render的节点！")
            raise ValueError("Value Error!")
        else:
            self.__objcm.objnp.reparentTo(objnp)
        # self.setMat(self.__objcm.getMat())
        # print(self.objbdb.gethomomat())
        self.__objcm.objnp.setMat(base.pg.np4ToMat4(self.objbdb.get_homomat()))

    def removeNode(self):
        """
        移除对象节点

        :return: None
        """
        self.__objcm.objnp.removeNode()
        base.physicsworld.remove(self.__objbdb)

    def detachNode(self):
        """
        分离对象节点

        :return: None
        """
        self.__objcm.objnp.detachNode()

    def showcn(self):
        """
        显示碰撞节点

        :return: None
        """
        # 重新连接以绕过deepcopy的失败
        self.__cdnp.removeNode()
        self.__cdnp = self.__objnp.attachNewNode(self.__cdcn)
        self.__cdnp.show()

    def showLocalFrame(self):
        """
        显示局部坐标系

        :return: None
        """
        self.__localframe = base.pggen.genAxis()
        self.__localframe.reparentTo(self.objnp)

    def unshowLocalFrame(self):
        """
        隐藏局部坐标系

        :return: None
        """
        if self.__localframe is not None:
            self.__localframe.removeNode()
            self.__localframe = None

    def unshowcn(self):
        """
        隐藏碰撞节点

        :return: None
        """
        self.__cdnp.hide()

    def copy(self):
        """
        复制当前对象

        :return: 新的BDModel对象
        """
        return BDModel(self)


if __name__ == "__main__":
    import os
    import numpy as np
    import basis.robot_math as rm
    import pandaplotutils.pandactrl as pc
    import random
    import basis

    base = pc.World(camp=[1000, 300, 1000], lookatpos=[0, 0, 0], toggledebug=False)
    base.setFrameRateMeter(True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    bunnycm = BDModel(objpath, mass=1, type="convex")

    objpath = os.path.join(basis.__path__[0], 'objects', 'bowlblock.stl')
    bunnycm2 = BDModel(objpath, mass=0, type="triangle", dynamic=False)
    bunnycm2.set_rgba(0, 0.7, 0.7, 1.0)
    # bunnycm2.reparentTo(base.render)
    bunnycm2.set_pos(0, 0, 0)
    base.attachRUD(bunnycm2)


    def update(bunnycm, task):
        if base.inputmgr.keyMap['space'] is True:
            for i in range(300):
                bunnycm1 = bunnycm.copy()
                bunnycm1.setMass(.1)
                # bunnycm1.setColor(0.7, 0, 0.7, 1.0)
                bunnycm1.setColor(random.random(), random.random(), random.random(), 1.0)
                # bunnycm1.reparentTo(base.render)
                # rotmat = rm.rodrigues([0,0,1], 15)
                rotmat = rm.rotmat_from_euler(0, 0, 15)
                z = math.floor(i / 100)
                y = math.floor((i - z * 100) / 10)
                x = i - z * 100 - y * 10
                print(x, y, z, "\n")
                bunnycm1.setMat(base.pg.npToMat4(rotmat, np.array([x * 15 - 70, y * 15 - 70, 150 + z * 15])))
                base.attachRUD(bunnycm1)
        base.inputmgr.keyMap['space'] = False
        return task.cont


    base.pggen.plotAxis(base.render)
    taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

    base.run()
