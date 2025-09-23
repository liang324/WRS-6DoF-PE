from direct.showbase.ShowBaseGlobal import globalClock
from direct.task.TaskManagerGlobal import taskMgr
import modeling.collision_model as cm
from panda3d.ode import OdeWorld, OdeSimpleSpace, OdeJointGroup
from panda3d.ode import OdeBody, OdeMass, OdeBoxGeom, OdePlaneGeom, OdeTriMeshGeom, OdeTriMeshData
from panda3d.core import BitMask32, CardMaker, Vec4
import visualization.panda.world as wd
from random import randint, random
import math
import basis.robot_math as rm
import basis.data_adapter as da
import numpy as np

base = wd.World(cam_pos=[15, 15, 15], lookat_pos=[0, 0, 0], toggle_debug=True)

world = OdeWorld()
world.setGravity(0, 0, -9.81)  # 设置重力加速度
world.setQuickStepNumIterations(100)  # 设置快速步进迭代次数
world.setErp(.2)  # 设置误差修正参数
world.setCfm(1e-3)  # 设置约束力混合参数

# 初始化表面属性表,用于自动碰撞
world.initSurfaceTable(1)
world.setSurfaceEntry(0, 0, 150, 0.0, 9.1, 0.9, 0.00001, 0.0, 0.002)

# 创建一个空间,并添加一个接触组用于添加接触关节
space = OdeSimpleSpace()
space.setAutoCollideWorld(world)
contactgroup = OdeJointGroup()
space.setAutoCollideJointGroup(contactgroup)

box = cm.gen_box(extent=[.3, .3, .3])

# 添加随机数量的立方体
boxes = []
for i in range(randint(5, 10)):
    # 设置几何体
    new_box = box.copy()
    new_box.set_pos(np.array([random() * 10 - 5, random() * 10 - 5, 1 + random()]))
    new_box.set_rgba([random(), random(), random(), 1])
    new_box.set_rotmat(rm.rotmat_from_euler(random() * math.pi / 4, random() * math.pi / 4, random() * math.pi / 4))
    new_box.attach_to(base)
    # 创建物体并设置质量
    boxBody = OdeBody(world)
    M = OdeMass()
    M.setBox(3, .3, .3, .3)
    boxBody.setMass(M)
    boxBody.setPosition(da.npv3_to_pdv3(new_box.get_pos()))
    boxBody.setQuaternion(da.npmat3_to_pdquat(new_box.get_rotmat()))
    # 创建一个 BoxGeom
    boxGeom = OdeBoxGeom(space, .3, .3, .3)
    # boxGeom = OdeTriMeshGeom(space, OdeTriMeshData(new_box.objpdnp, True))
    boxGeom.setCollideBits(BitMask32(0x00000002))
    boxGeom.setCategoryBits(BitMask32(0x00000001))
    boxGeom.setBody(boxBody)
    boxes.append((new_box, boxBody))

# 添加一个平面用于碰撞
ground = cm.gen_box(extent=[20, 20, 1], rgba=[.3, .3, .3, 1])
ground.set_pos(np.array([0, 0, -1.5]))
ground.attach_to(base)
# groundGeom = OdeTriMeshGeom(space, OdeTriMeshData(ground.objpdnp, True))
groundGeom = OdePlaneGeom(space, Vec4(0, 0, 1, -1))
groundGeom.setCollideBits(BitMask32(0x00000001))
groundGeom.setCategoryBits(BitMask32(0x00000002))


# 模拟任务
def simulationTask(task):
    space.autoCollide()  # 设置接触关节
    # 进行模拟步进并设置新位置
    world.step(globalClock.getDt())
    for cmobj, body in boxes:
        cmobj.set_homomat(rm.homomat_from_posrot(da.npv3_to_pdv3(body.getPosition()),
                                                 da.pdmat3_to_npmat3(body.getRotation())))
    contactgroup.empty()  # 清除接触关节
    return task.cont


# 等待一小段时间,然后开始模拟
taskMgr.doMethodLater(0.5, simulationTask, "Physics Simulation")

base.run()
