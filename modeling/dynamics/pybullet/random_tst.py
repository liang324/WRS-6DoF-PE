# 这段代码创建了一个简单的物理仿真场景,加载了一个平面和一个 R2D2 机器人模型,并运行仿真 10000 步.
# 在仿真结束后,获取并打印 R2D2 的最终位置和方向.仿真是在图形界面中进行的,因此可以看到模型的运动.

import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI, options="--opengl2")  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
