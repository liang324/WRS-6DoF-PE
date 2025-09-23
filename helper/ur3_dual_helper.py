import numpy as np
import robot_sim.robots.ur3_dual.ur3_dual as ur3ds
import robot_con.ur.ur3_dual_x as ur3dx
import motion.probabilistic.rrt_connect as rrtc
import motion.optimization_based.incremental_nik as inik
import manipulation.pick_place_planner as ppp
import visualization.panda.world as wd


class UR3DualHelper(object):
    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 use_real=False,
                 create_sim_world=True,
                 lft_robot_ip='10.2.0.50',
                 rgt_robot_ip='10.2.0.51',
                 pc_ip='10.2.0.100',
                 cam_pos=np.array([2, 1, 3]),
                 lookat_pos=np.array([0, 0, 1.1]),
                 auto_cam_rotate=False):

        """
        初始化 UR3DualHelper 类,用于管理 UR3 双臂机器人

        :param pos: 机器人的初始位置
        :param rotmat: 机器人的初始旋转矩阵
        :param use_real: 是否使用真实机器人
        :param create_sim_world: 是否创建模拟世界
        :param lft_robot_ip: 左臂机器人的 IP 地址
        :param rgt_robot_ip: 右臂机器人的 IP 地址
        :param pc_ip: 控制 PC 的 IP 地址
        :param cam_pos: 相机位置
        :param lookat_pos: 相机观察目标位置
        :param auto_cam_rotate: 是否自动旋转相机
        """
        self.robot_s = ur3ds.UR3Dual(pos=pos, rotmat=rotmat)
        self.rrt_planner = rrtc.RRTConnect(self.robot_s)
        self.inik_solver = inik.IncrementalNIK(self.robot_s)
        self.pp_planner = ppp.PickPlacePlanner(self.robot_s)
        if use_real:
            self.robot_x = ur3dx.UR3DualX(lft_robot_ip=lft_robot_ip,
                                          rgt_robot_ip=rgt_robot_ip,
                                          pc_ip=pc_ip)
        if create_sim_world:
            self.sim_world = wd.World(cam_pos=cam_pos,
                                      lookat_pos=lookat_pos,
                                      auto_cam_rotate=auto_cam_rotate)

    def plan_motion(self,
                    component_name,
                    start_conf,
                    goal_conf,
                    obstacle_list=[],
                    otherrobot_list=[],
                    ext_dist=2,
                    maxiter=1000,
                    maxtime=15.0,
                    animation=False):
        """
        规划机器人的运动路径

        :param component_name: 机器人组件名称
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param ext_dist: 扩展距离
        :param maxiter: 最大迭代次数
        :param maxtime: 最大规划时间
        :param animation: 是否显示动画
        :return: 规划的路径
        """
        path = self.rrt_planner.plan(component_name=component_name,
                                     start_conf=start_conf,
                                     goal_conf=goal_conf,
                                     obstacle_list=obstacle_list,
                                     otherrobot_list=otherrobot_list,
                                     ext_dist=ext_dist,
                                     max_iter=maxiter,
                                     max_time=maxtime,
                                     animation=animation)
        return path

    def plan_pick_and_place(self,
                            manipulator_name,
                            hand_name,
                            objcm,
                            grasp_info_list,
                            start_conf,
                            goal_homomat_list):
        """
        规划抓取和放置任务

        :param manipulator_name: 操作器名称
        :param hand_name: 手部名称
        :param objcm: 目标对象
        :param grasp_info_list: 抓取信息列表
        :param start_conf: 起始配置
        :param goal_homomat_list: 目标齐次矩阵列表
        :return: None

        author: weiwei
        date: 20210409
        """
        self.pp_planner.gen_pick_and_place_motion(manipulator_name,
                                                  hand_name,
                                                  objcm,
                                                  grasp_info_list,
                                                  start_conf,
                                                  goal_homomat_list)
