import numpy as np
import rrt_connect
import robot_sim.robots.xybot.xybot as xyb
import time

obstacle_list = [
    ((5, 5), 3),
    ((3, 6), 3),
    ((3, 8), 3),
    ((3, 10), 3),
    ((7, 5), 3),
    ((9, 5), 3),
    ((10, 5), 3),
    ((10, 0), 3),
    ((10, -2), 3),
    ((10, -4), 3),
    ((15, 5), 3),
    ((15, 7), 3),
    ((15, 9), 3),
    ((15, 11), 3),
    ((0, 12), 3),
    ((-1, 10), 3),
    ((-2, 8), 3)
]

robot = xyb.XYBot()
rrt_s = rrt_connect.RRTConnect(robot)

start_conf = np.array([15, 0])
goal_conf = np.array([5, -2.5])

total_t = 0
tic = time.time()
path = rrt_s.plan(component_name='all',
                  start_conf=start_conf,
                  goal_conf=goal_conf,
                  obstacle_list=obstacle_list,
                  ext_dist=1,
                  rand_rate=70,
                  max_iter=100000,
                  max_time=1000,
                  smoothing_iterations=0,
                  animation=False)
toc = time.time()
total_t = toc - tic
print("rrt costs:", total_t)
