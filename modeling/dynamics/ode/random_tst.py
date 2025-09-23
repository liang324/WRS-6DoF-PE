from direct.showbase.ShowBaseGlobal import globalClock
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
from panda3d.ode import OdeWorld, OdeBody, OdeMass, OdeBallJoint
import modeling.collision_model as cm
import modeling.geometric_model as gm
import basis.data_adapter as da
import numpy as np

base = wd.World(cam_pos=[7, 0, 0], lookat_pos=[0, 0, -.5], toggle_debug=True)
radius = .1  # 球体半径

# 创建第一个球体,设置位置和颜色,并将其附加到世界中
sphere_a = cm.gen_sphere(radius=radius)
sphere_a.set_pos([0, .3, -.3])
sphere_a.set_rgba([1, .2, .3, 1])
sphere_a.attach_to(base)

# 创建第二个球体,设置位置和颜色,并将其附加到世界中
sphere_b = cm.gen_sphere(radius=radius)
sphere_b.set_pos([0, 1.25, -.7])
sphere_b.set_rgba([.3, .2, 1, 1])
sphere_b.attach_to(base)

# 生成线段连接球体和原点,并附加到世界中
gm.gen_linesegs([[np.zeros(3), sphere_a.get_pos()]], thickness=.05, rgba=[0, 1, 0, 1]).attach_to(base)
gm.gen_linesegs([[sphere_a.get_pos(), sphere_b.get_pos()]], thickness=.05, rgba=[0, 0, 1, 1]).attach_to(base)


# 设置物理世界和重力
world = OdeWorld()
world.setGravity(0, 0, -9.81)

# 创建球体 A 的物理体,设置质量和位置
body_sphere_a = OdeBody(world)
M = OdeMass()
M.setSphere(7874, radius)
body_sphere_a.setMass(M)
body_sphere_a.setPosition(da.npv3_to_pdv3(sphere_a.get_pos()))


# 创建球体 B 的物理体,设置质量和位置
body_sphere_b = OdeBody(world)
M = OdeMass()
M.setSphere(7874, radius)
body_sphere_b.setMass(M)
body_sphere_b.setPosition(da.npv3_to_pdv3(sphere_b.get_pos()))


# 创建关节,将球体 A 附加到环境
earth_a_jnt = OdeBallJoint(world)
earth_a_jnt.attach(body_sphere_a, None)  # 附加到环境
earth_a_jnt.setAnchor(0, 0, 0)


# 创建关节,将球体 A 和球体 B 连接
earth_b_jnt = OdeBallJoint(world)
earth_b_jnt.attach(body_sphere_a, body_sphere_b)
earth_b_jnt.setAnchor(0, .3, -.3)


# 创建一个累加器来跟踪模拟运行的时间
deltaTimeAccumulator = 0.0
# 设置模拟步长,使模拟以每秒 90 帧的速度运行
stepSize = 1.0 / 90.0


# 模拟任务
def simulationTask(task):
    # 进行模拟步骤并设置新位置
    world.quickStep(globalClock.getDt())
    sphere_a.set_pos(da.pdv3_to_npv3(body_sphere_a.getPosition()))
    sphere_b.set_pos(da.pdv3_to_npv3(body_sphere_b.getPosition()))
    gm.gen_linesegs([[np.zeros(3), sphere_a.get_pos()]], thickness=.05, rgba=[0, 1, 0, 1]).attach_to(base)
    gm.gen_linesegs([[sphere_a.get_pos(), sphere_b.get_pos()]], thickness=.05, rgba=[0, 0, 1, 1]).attach_to(base)
    return task.cont


taskMgr.doMethodLater(1.0, simulationTask, "Physics Simulation")
base.run()
