from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletConvexHullShape, BulletBoxShape
from panda3d.core import TransformState, Vec3, GeomVertexRewriter, CollisionBox, Point3
import copy
import modeling.geometric_model as gm
import basis.data_adapter as dh
import basis.robot_math as rm
import numpy as np


class BDBody(BulletRigidBodyNode):
    def __init__(self,
                 initor,
                 cdtype="triangles",
                 mass=.3,
                 restitution=0,
                 allow_deactivation=False,
                 allow_ccd=True,
                 friction=.2,
                 dynamic=True,
                 physics_scale=1e3,
                 name="rbd"):
        """
        BDBody 类用于创建一个 Bullet 刚体节点,并根据给定的参数初始化其物理属性

        注意:在 GitHub 版本(20210418)中,三角形碰撞检测似乎不起作用(非常慢)
        如果可能,使用凸形

        :param initor: 可以是自身(用于复制),或碰撞模型的实例
        :param cdtype: 碰撞检测类型,"triangles" 或 "convex" 或 "box"
        :param mass: 质量
        :param restitution: 弹性恢复系数
        :param allow_deactivation: 是否允许去激活
        :param allow_ccd: 是否允许连续碰撞检测
        :param friction: 摩擦系数
        :param dynamic: 仅适用于三角形类型,如果对象不随力移动,则不是动态的
        :param name: 节点名称

        author: weiwei
        date: 20190626, 20201119
        """
        super().__init__(name)
        self.physics_scale = physics_scale
        if isinstance(initor, gm.GeometricModel):
            if initor._objtrm is None:
                raise ValueError("仅适用于具有三角网格的模型！")

            # 计算中心质量并缩放
            self.com = initor.objtrm.center_mass * self.physics_scale
            self.setMass(mass)
            self.setRestitution(restitution)
            self.setFriction(friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)

            if allow_deactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(.01 * self.physics_scale)
                self.setAngularSleepThreshold(.01 * self.physics_scale)
            else:
                self.setDeactivationEnabled(False)
            if allow_ccd:  # 连续碰撞检测
                self.setCcdMotionThreshold(1e-7)
                self.setCcdSweptSphereRadius(0.0005 * self.physics_scale)

            # 获取几何体并进行深拷贝
            geom_np = initor.objpdnp.getChild(0).find("+GeomNode")
            geom = copy.deepcopy(geom_np.node().getGeom(0))
            vdata = geom.modifyVertexData()
            vertices = copy.deepcopy(np.frombuffer(vdata.modifyArrayHandle(0).getData(), dtype=np.float32))
            vertices.shape = (-1, 6)
            vertices[:, :3] = vertices[:, :3] * self.physics_scale - self.com
            vdata.modifyArrayHandle(0).setData(vertices.astype(np.float32).tobytes())
            geomtf = geom_np.getTransform()
            geomtf = geomtf.setPos(geomtf.getPos() * self.physics_scale)
            if cdtype == "triangles":
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
            elif cdtype == 'box':
                minx = min(vertices[:, 0])
                miny = min(vertices[:, 1])
                minz = min(vertices[:, 2])
                maxx = max(vertices[:, 0])
                maxy = max(vertices[:, 1])
                maxz = max(vertices[:, 2])
                pcd_box = CollisionBox(Point3(minx, miny, minz), Point3(maxx, maxy, maxz))
                bulletshape = BulletBoxShape.makeFromSolid(pcd_box)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            elif cdtype == 'sphere':
                # sphere shape
                minx = min(vertices[:, 0])
                miny = min(vertices[:, 1])
                minz = min(vertices[:, 2])
                maxx = max(vertices[:, 0])
                maxy = max(vertices[:, 1])
                maxz = max(vertices[:, 2])
                radius = max([maxx - minx, maxy - miny, maxz - minz]) / 2
                bulletshape = BulletSphereShape(radius)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            else:
                raise NotImplementedError

            # 设置变换矩阵
            pd_homomat = geomtf.getMat()
            pd_com_pos = pd_homomat.xformPoint(Vec3(self.com[0], self.com[1], self.com[2]))
            np_homomat = dh.pdmat4_to_npmat4(pd_homomat)
            np_com_pos = dh.pdv3_to_npv3(pd_com_pos)
            np_homomat[:3, 3] = np_com_pos  # 更新中心到质心
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(np_homomat)))
        elif isinstance(initor, BDBody):
            # 复制已有的 BDBody 实例
            self.com = initor.com.copy()
            self.setMass(initor.getMass())
            self.setRestitution(initor.restitution)
            self.setFriction(initor.friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allow_deactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(.01 * self.physics_scale)
                self.setAngularSleepThreshold(.01 * self.physics_scale)
            else:
                self.setDeactivationEnabled(False)
            if allow_ccd:
                self.setCcdMotionThreshold(1e-7)
                self.setCcdSweptSphereRadius(0.0005 * self.physics_scale)
            np_homomat = copy.deepcopy(initor.get_homomat())
            np_homomat[:3, 3] = np_homomat[:3, 3] * self.physics_scale
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(np_homomat)))
            self.addShape(initor.getShape(0), initor.getShapeTransform(0))

    def get_pos(self):
        """
        获取物体的位置

        :return: 物体的位置向量
        """
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        pos = dh.pdv3_to_npv3(pdv3) / self.physics_scale
        return pos

    def set_pos(self, npvec3):
        """
        设置物体的位置

        :param npvec3: 位置向量
        """
        self.setPos(dh.pdv3_to_npv3(npvec3) * self.physics_scale)

    def get_homomat(self):
        """
        获取考虑原始局部框架的齐次变换矩阵

        动态物体在质心定义的局部框架中移动,而不是返回动态物体的齐次变换矩阵,
        该方法返回原始局部框架的姿态,返回的齐次变换矩阵可用于碰撞体的渲染.
        :return: 齐次变换矩阵

        :return:
        author: weiwei
        date: 2019?, 20201119
        """
        pd_homomat = self.getTransform().getMat()
        pd_com_pos = pd_homomat.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        np_homomat = dh.pdmat4_to_npmat4(pd_homomat)
        np_com_pos = dh.pdv3_to_npv3(pd_com_pos)
        np_homomat[:3, 3] = np_com_pos / self.physics_scale
        return np_homomat

    def set_homomat(self, homomat):
        """
        设置 dynamic 物体的姿态

        :param homomat: 原始框架的齐次变换矩阵(碰撞模型)
        :return: None

        author: weiwei
        date: 2019?, 20201119
        """
        tmp_homomat = copy.deepcopy(homomat)
        tmp_homomat[:3, 3] = tmp_homomat[:3, 3] * self.physics_scale
        pos = rm.homomat_transform_points(tmp_homomat, self.com)
        rotmat = tmp_homomat[:3, :3]
        self.setTransform(TransformState.makeMat(dh.npv3mat3_to_pdmat4(pos, rotmat)))

    def copy(self):
        """
        复制当前 BDBody 实例

        :return: 新的 BDBody 实例
        """
        return BDBody(self)
