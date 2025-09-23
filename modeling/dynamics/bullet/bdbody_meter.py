from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.core import TransformState, Vec3, GeomVertexRewriter
import copy
import modeling.geometric_model as gm
import basis.data_adapter as dh
import basis.robot_math as rm


class BDBody(BulletRigidBodyNode):
    def __init__(self, objinit, cdtype="triangle", mass=.3, restitution=0, allowdeactivation=False, allowccd=True,
                 friction=.2, dynamic=True, name="rbd"):
        """
        BDBody 类用于创建一个 Bullet 刚体节点,并根据给定的参数初始化其物理属性

        :param objinit: 可以是自身(用于复制),或碰撞模型的实例
        :param cdtype: 碰撞检测类型,"triangle" 或 "convex"
        :param mass: 质量
        :param restitution: 弹性恢复系数
        :param friction: 摩擦系数
        :param dynamic: 仅适用于三角形类型,如果对象不随力移动,则不是动态的
        :param name: 节点名称

        author: weiwei
        date: 20190626, 20201119
        """
        super().__init__(name)
        if isinstance(objinit, gm.GeometricModel):
            if objinit._objtrm is None:
                raise ValueError("仅适用于具有三角网格的模型！")

            # 计算中心质量
            self.com = objinit.objtrm.center_mass
            self.setMass(mass)
            self.setRestitution(restitution)
            self.setFriction(friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allowdeactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(0.001)
                self.setAngularSleepThreshold(0.001)
            else:
                self.setDeactivationEnabled(False)
            if allowccd:  # 连续碰撞检测
                self.setCcdMotionThreshold(1e-6)
                self.setCcdSweptSphereRadius(0.0005)

            # 获取几何体并进行深拷贝
            gnd = objinit.objpdnp.getChild(0).find("+GeomNode")
            geom = copy.deepcopy(gnd.node().getGeom(0))
            vdata = geom.modifyVertexData()
            vertrewritter = GeomVertexRewriter(vdata, 'vertex')
            while not vertrewritter.isAtEnd():  # 将局部坐标移至几何体以正确更新动态变化
                v = vertrewritter.getData3f()
                vertrewritter.setData3f(v[0] - self.com[0], v[1] - self.com[1], v[2] - self.com[2])
            geomtf = gnd.getTransform()
            if cdtype == "triangle":
                geombmesh = BulletTriangleMesh()
                geombmesh.addGeom(geom)
                bulletshape = BulletTriangleMeshShape(geombmesh, dynamic=dynamic)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            elif cdtype == "convex":
                bulletshape = BulletConvexHullShape()  # TODO: 计算凸包？
                bulletshape.addGeom(geom, geomtf)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            else:
                raise NotImplementedError

            # 设置变换矩阵
            pdmat4 = geomtf.getMat()
            pdv3 = pdmat4.xformPoint(Vec3(self.com[0], self.com[1], self.com[2]))
            homomat = dh.pdmat4_to_npmat4(pdmat4)
            pos = dh.pdv3_to_npv3(pdv3)
            homomat[:3, 3] = pos  # 更新中心到质心
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(homomat)))
        elif isinstance(objinit, BDBody):
            # 复制已有的 BDBody 实例
            self.com = objinit.com.copy()
            self.setMass(objinit.getMass())
            self.setRestitution(objinit.restitution)
            self.setFriction(objinit.friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allowdeactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(0.001)
                self.setAngularSleepThreshold(0.001)
            else:
                self.setDeactivationEnabled(False)
            if allowccd:
                self.setCcdMotionThreshold(1e-6)
                self.setCcdSweptSphereRadius(0.0005)
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(objinit.gethomomat())))
            self.addShape(objinit.getShape(0), objinit.getShapeTransform(0))

    def getpos(self):
        """
        获取物体的位置

        :return: 1x3 的 numpy 数组
        """
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        # homomat = dh.pdmat4_to_npmat4(pdmat4)
        pos = dh.pdv3_to_npv3(pdv3)
        return pos

    def setpos(self, npvec3):
        """
        设置物体的位置
        :param npvec3: 1x3 的 numpy 数组
        """
        self.setPos(dh.pdv3_to_npv3(npvec3))

    def gethomomat(self):
        """
        获取考虑原始局部框架的齐次变换矩阵

        动态物体在质心定义的局部框架中移动,而不是返回动态物体的齐次变换矩阵,
        该方法返回原始局部框架的姿态,返回的齐次变换矩阵可用于碰撞体的渲染.
        :return: 齐次变换矩阵

        author: weiwei
        date: 2019?, 20201119
        """
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        homomat = dh.pdmat4_to_npmat4(pdmat4)
        pos = dh.pdv3_to_npv3(pdv3)
        homomat[:3, 3] = pos
        return homomat

    def sethomomat(self, homomat):
        """
        设置动态物体的姿态

        :param homomat: 原始框架的齐次变换矩阵(碰撞模型)
        :return:

        author: weiwei
        date: 2019?, 20201119
        """
        pos = rm.homomat_transform_points(homomat, self.com)
        rotmat = homomat[:3, :3]
        self.setTransform(TransformState.makeMat(dh.npv3mat3_to_pdmat4(pos, rotmat)))

    def setmass(self, mass):
        """
        设置物体的质量

        :param mass: 质量值
        """
        self.mass = mass
        self.setMass(mass)

    def copy(self):
        """
        复制当前 BDBody 实例

        :return: 新的 BDBody 实例
        """
        return BDBody(self)
