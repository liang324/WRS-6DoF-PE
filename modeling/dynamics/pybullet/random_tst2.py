# 这段代码设置了一个简单的物理仿真环境,应用了重力,并以 240 Hz 的频率运行仿真 10000 步.
# 由于使用了 p.DIRECT 模式,仿真是在后台运行的,没有图形界面显示.


import pybullet as p
import time

p_client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
