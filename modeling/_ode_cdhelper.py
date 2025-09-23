import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil, OdeRayGeom

import basis.data_adapter as da
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
from panda3d.core import CollisionBox, CollisionSphere, CollisionPolygon, GeomVertexReader


# util functions
def gen_cdmesh_vvnf(vertices, vertex_normals, faces):
    """
    根据顶点、顶点法线和面生成碰撞检测网格(cdmesh)

    :param vertices: 顶点列表
    :param vertex_normals: 顶点法线列表
    :param faces: 面列表
    :return: panda3d.ode.OdeTriMeshGeom 对象

    author: weiwei
    date: 20210118
    """
    # 从顶点、法线和面生成 Panda3D 节点路径
    objpdnp = da.nodepath_from_vvnf(vertices, vertex_normals, faces)
    # 创建 OdeTriMeshGeom 对象用于碰撞检测
    obj_ot_geom = OdeTriMeshGeom(OdeTriMeshData(objpdnp, True))
    return obj_ot_geom


def copy_cdmesh(obj_ot_geom):
    """
    复制一个 OdeTriMeshGeom 对象

    :param obj_ot_geom: 要复制的 OdeTriMeshGeom 对象
    :return: 复制的 OdeTriMeshGeom 对象
    """
    # 使用原始对象的网格数据创建新的 OdeTriMeshGeom 对象
    obj_ot_geom_copy = OdeTriMeshGeom(obj_ot_geom.getTriMeshData())
    return obj_ot_geom_copy


# def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
#     """
#     generate a plane bulletrigidbody node
#     :param updirection: the normal parameter of bulletplaneshape at panda3d
#     :param offset: the d parameter of bulletplaneshape at panda3d
#     :param name:
#     :return: bulletrigidbody
#     author: weiwei
#     date: 20170202, tsukuba
#     """
#     bulletplnode = BulletRigidBodyNode(name)
#     bulletplshape = BulletPlaneShape(Vec3(updirection[0], updirection[1], updirection[2]), offset)
#     bulletplshape.setMargin(0)
#     bulletplnode.addShape(bulletplshape)
#     return bulletplnode

def is_collided(obj_ode_trimesh0, obj_ode_trimesh1):
    """
    检查两个 OdeTriMeshGeom 对象是否发生碰撞

    胡正涛老师版本

    :param obj_ode_trimesh0: 第一个 OdeTriMeshGeom 对象
    :param obj_ode_trimesh1: 第二个 OdeTriMeshGeom 对象
    :return: 碰撞结果和接触点列表

    author: weiwei
    date: 20210118, 20211215
    """
    # 使用 OdeUtil.collide 方法进行碰撞检测
    contact_entry = OdeUtil.collide(obj_ode_trimesh0, obj_ode_trimesh1, max_contacts=10)
    # 将接触点转换为 NumPy 向量
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    # 返回碰撞结果和接触点列表
    return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)


# def is_collided(objcm0, objcm1):
#     """
#     检查两个碰撞模型是否发生碰撞,并返回碰撞状态和碰撞点
#
#     :param objcm0: 碰撞模型或碰撞模型集合的实例
#     :param objcm1: 碰撞模型或碰撞模型集合的实例
#     :return:
#
#     author: weiwei
#     date: 20210118
#     """
#     obj0 = gen_cdmesh_vvnf(*objcm0.extract_rotated_vvnf())
#     obj1 = gen_cdmesh_vvnf(*objcm1.extract_rotated_vvnf())
#     contact_entry = OdeUtil.collide(obj0, obj1, max_contacts=10)
#     contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
#     return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)


def update_pose(obj_ode_trimesh, objnp):
    """
    使用 objnp 的位置和四元数更新 obj_ode_trimesh 的姿态

    :param obj_ode_trimesh: OdeTriMeshGeom 对象
    :param objnp: Panda3D 节点路径对象
    :return: None

    author: weiwei
    date: 20211215
    """
    obj_ode_trimesh.setPosition(objnp.getPos())  # 设置碰撞网格的位置
    obj_ode_trimesh.setQuaternion(objnp.getQuat())  # 设置碰撞网格的旋转(四元数)


def rayhit_closet(pfrom, pto, objcm):
    """
    计算从 pfrom 到 pto 的射线与 objcm 的碰撞模型的最近碰撞点和法线

    :param pfrom: 射线起点
    :param pto: 射线终点
    :param objcm: 碰撞模型对象
    :return: 最近的碰撞点和法线

    author: weiwei
    date: 20190805
    """
    # 生成目标碰撞网格
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    # 创建射线几何体
    ray = OdeRayGeom(length=1)
    # 计算射线的方向和长度
    length, dir = rm.unit_vector(pto - pfrom, toggle_length=True)
    # 设置射线的起点和方向
    ray.set(pfrom[0], pfrom[1], pfrom[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    # 执行碰撞检测
    contact_entry = OdeUtil.collide(ray, tgt_cdmesh, max_contacts=10)
    # 获取接触点
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    # 找到最近的接触点
    min_id = np.argmin(np.linalg.norm(pfrom - np.array(contact_points), axis=1))
    # 获取接触法线
    contact_normals = [da.pdv3_to_npv3(contact_entry.getContactGeom(i).getNormal()) for i in
                       range(contact_entry.getNumContacts())]
    return contact_points[min_id], contact_normals[min_id]


def rayhit_all(pfrom, pto, objcm):
    """
    计算从 pfrom 到 pto 的射线与 objcm 的碰撞模型的所有碰撞点和法线

    :param pfrom: 射线起点
    :param pto: 射线终点
    :param objcm: 碰撞模型对象
    :return: 所有碰撞点和法线

    author: weiwei
    date: 20190805
    """
    # 生成目标碰撞网格
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    # 创建射线几何体
    ray = OdeRayGeom(length=1)
    # 计算射线的方向和长度
    length, dir = rm.unit_vector(pto - pfrom, toggle_length=True)
    # 设置射线的起点和方向
    ray.set(pfrom[0], pfrom[1], pfrom[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    # 执行碰撞检测
    hit_entry = OdeUtil.collide(ray, tgt_cdmesh)
    # 获取所有接触点
    hit_points = [da.pdv3_to_npv3(point) for point in hit_entry.getContactPoints()]
    # 获取所有接触法线
    hit_normals = [da.pdv3_to_npv3(hit_entry.getContactGeom(i).getNormal()) for i in range(hit_entry.getNumContacts())]
    return hit_points, hit_normals


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm1 = cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1, 1, .3, .2])

    objcm2 = objcm1.copy()
    objcm2.set_pos(objcm1.get_pos() + np.array([.05, .02, .0]))
    objcm1.change_cdmesh_type('convex_hull')
    objcm2.change_cdmesh_type('obb')
    iscollided, contact_points = is_collided(objcm1, objcm2)
    # iscollided, contact_points = is_collided(objcm1, objcm2, toggle_contact_points=True)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)

    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.001).attach_to(base)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -1.0])
    pto = np.array([0, 0, 0]) + np.array([0.02, 0.02, 0.02])
    # pfrom = np.array([0, 0, 0]) + np.array([0.0, 0.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([0.0, 0.0, -1.0])
    # hit_point, hit_normal = rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm1)
    hit_points, hit_normals = rayhit_all(pfrom=pfrom, pto=pto, objcm=objcm1)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(type='box')
    # objcm.show_cdmesh(type='convex_hull')
    # for hitpos, hitnormal in zip([hit_point], [hit_normal]):
    for hitpos, hitnormal in zip(hit_points, hit_normals):
        gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
        gm.gen_arrow(hitpos, epos=hitpos + hitnormal * .03, thickness=.002, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
