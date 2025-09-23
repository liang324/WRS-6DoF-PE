import os
import math
import copy
import pickle
import numpy as np
import basis.data_adapter as da
import modeling.collision_model as cm
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc


class ADPlanner(object):  # AD = Approach_Depart
    def __init__(self, robot_s):
        """
        初始化 ADPlanner 类

        :param robot_s: 机器人实例
        author: weiwei, hao
        date: 20191122, 20210113
        """
        self.robot_s = robot_s
        # 初始化增量逆运动学求解器
        self.inik_slvr = inik.IncrementalNIK(self.robot_s)
        # 初始化 RRT-Connect 路径规划器
        self.rrtc_planner = rrtc.RRTConnect(self.robot_s)

    def gen_jawwidth_motion(self, conf_list, jawwidth):
        """
        生成夹爪宽度运动列表
        :param conf_list: 配置列表
        :param jawwidth: 夹爪宽度
        :return: 夹爪宽度列表
        """
        jawwidth_list = []
        for _ in conf_list:
            jawwidth_list.append(jawwidth)
        return jawwidth_list

    def gen_approach_linear(self,
                            component_name,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            approach_direction=None,
                            approach_distance=.1,
                            approach_jawwidth=.05,
                            granularity=0.03,
                            obstacle_list=[],
                            seed_jnt_values=None,
                            toggle_end_grasp=False,
                            end_jawwidth=.0):
        """
        生成线性接近路径
        :param component_name: 组件名称
        :param goal_tcp_pos: 目标 TCP 位置
        :param goal_tcp_rotmat: 目标 TCP 旋转矩阵
        :param approach_direction: 接近方向,默认为旋转矩阵的 z 轴
        :param approach_distance: 接近距离
        :param approach_jawwidth: 接近时的夹爪宽度
        :param granularity: 步长
        :param obstacle_list: 障碍物列表
        :param seed_jnt_values: 初始关节值
        :param toggle_end_grasp: 是否在末尾添加不同夹爪宽度的配置
        :param end_jawwidth: 末尾的夹爪宽度,仅在 toggle_end_grasp 为 True 时使用
        :return: 配置列表和夹爪宽度列表
        author: weiwei
        date: 20210125
        """
        if approach_direction is None:
            # 如果未提供接近方向,默认为旋转矩阵的 z 轴
            approach_direction = goal_tcp_rotmat[:, 2]

        # 生成相对线性运动的配置列表
        conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                         goal_tcp_pos,
                                                         goal_tcp_rotmat,
                                                         approach_direction,
                                                         approach_distance,
                                                         obstacle_list=obstacle_list,
                                                         granularity=granularity,
                                                         type='sink',
                                                         seed_jnt_values=seed_jnt_values)
        if conf_list is None:
            print('无法执行线性接近!')
            return None, None
        else:
            if toggle_end_grasp:
                # 如果需要在末尾添加不同夹爪宽度的配置
                jawwidth_list = self.gen_jawwidth_motion(conf_list, approach_jawwidth)
                conf_list += [conf_list[-1]]
                jawwidth_list += [end_jawwidth]
                return conf_list, jawwidth_list
            else:
                return conf_list, self.gen_jawwidth_motion(conf_list, approach_jawwidth)

    def gen_depart_linear(self,
                          component_name,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          depart_direction=None,  # np.array([0, 0, 1])
                          depart_distance=.1,
                          depart_jawwidth=.05,
                          granularity=0.03,
                          obstacle_list=[],
                          seed_jnt_values=None,
                          toggle_begin_grasp=False,
                          begin_jawwidth=.0):
        """
        生成线性离开路径

        :param component_name: 组件名称
        :param start_tcp_pos: 起始 TCP 位置
        :param start_tcp_rotmat: 起始 TCP 旋转矩阵
        :param depart_direction: 离开方向,默认为旋转矩阵的负 z 轴
        :param depart_distance: 离开距离
        :param depart_jawwidth: 离开时的夹爪宽度
        :param granularity: 步长
        :param obstacle_list: 障碍物列表
        :param seed_jnt_values: 初始关节值
        :param toggle_begin_grasp: 是否在开始时添加不同夹爪宽度的配置
        :param begin_jawwidth: 开始时的夹爪宽度,仅在 toggle_begin_grasp 为 True 时使用
        :return: 配置列表、夹爪宽度列表

        author: weiwei
        date: 20210125
        """
        if depart_direction is None:
            # 如果未提供离开方向,默认为旋转矩阵的负 z 轴
            depart_direction = -start_tcp_rotmat[:, 2]

        # 生成相对线性运动的配置列表
        conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                         start_tcp_pos,
                                                         start_tcp_rotmat,
                                                         depart_direction,
                                                         depart_distance,
                                                         obstacle_list=obstacle_list,
                                                         granularity=granularity,
                                                         type='source',
                                                         seed_jnt_values=seed_jnt_values)
        if conf_list is None:
            print('无法执行离开动作！Cannot perform depart action!')
            return None, None
        else:
            if toggle_begin_grasp:
                # 如果需要在开始时添加不同夹爪宽度的配置
                jawwidth_list = self.gen_jawwidth_motion(conf_list, depart_jawwidth)
                conf_list = [conf_list[0]] + conf_list
                jawwidth_list = [begin_jawwidth] + jawwidth_list
                return conf_list, jawwidth_list
            else:
                return conf_list, self.gen_jawwidth_motion(conf_list, depart_jawwidth)

    def gen_approach_and_depart_linear(self,
                                       component_name,
                                       goal_tcp_pos,
                                       goal_tcp_rotmat,
                                       approach_direction=None,  # np.array([0, 0, -1])
                                       approach_distance=.1,
                                       approach_jawwidth=.05,
                                       depart_direction=None,  # np.array([0, 0, 1])
                                       depart_distance=.1,
                                       depart_jawwidth=0,
                                       granularity=.03,
                                       obstacle_list=[],
                                       seed_jnt_values=None):
        """
        生成线性接近和离开路径

        :param component_name: 组件名称
        :param goal_tcp_pos: 目标 TCP 位置
        :param goal_tcp_rotmat: 目标 TCP 旋转矩阵
        :param approach_direction: 接近方向,默认为旋转矩阵的 z 轴
        :param approach_distance: 接近距离
        :param approach_jawwidth: 接近时的夹爪宽度
        :param depart_direction: 离开方向,默认为旋转矩阵的 z 轴
        :param depart_distance: 离开距离
        :param depart_jawwidth: 离开时的夹爪宽度
        :param granularity: 步长
        :return: 接近配置列表、离开夹爪宽度列表 approach_conf_list, depart_jawwidth_list
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        if approach_direction is None:
            # 如果未提供接近方向,默认为旋转矩阵的 z 轴
            approach_direction = goal_tcp_rotmat[:, 2]

        # 生成接近路径的配置列表
        approach_conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                                  goal_tcp_pos,
                                                                  goal_tcp_rotmat,
                                                                  approach_direction,
                                                                  approach_distance,
                                                                  obstacle_list=obstacle_list,
                                                                  granularity=granularity,
                                                                  type='sink',
                                                                  seed_jnt_values=seed_jnt_values)

        # if approach_distance != 0 and len(approach_conf_list) == 0:
        if approach_conf_list is None or (approach_distance != 0 and len(approach_conf_list) == 0):
            print('无法执行接近动作！Cannot perform approach action!')
        else:
            if depart_direction is None:
                # 如果未提供离开方向,默认为旋转矩阵的 z 轴
                depart_direction = goal_tcp_rotmat[:, 2]
            # 生成离开路径的配置列表
            depart_conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                                    goal_tcp_pos,
                                                                    goal_tcp_rotmat,
                                                                    depart_direction,
                                                                    depart_distance,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=granularity,
                                                                    type='source',
                                                                    seed_jnt_values=approach_conf_list[-1])
            if depart_distance != 0 and len(depart_conf_list) == 0:
                print('无法执行离开动作！Cannot perform depart action!')
            else:
                # 生成接近和离开的夹爪宽度列表
                approach_jawwidth_list = self.gen_jawwidth_motion(approach_conf_list, approach_jawwidth)
                depart_jawwidth_list = self.gen_jawwidth_motion(depart_conf_list, depart_jawwidth)
                return approach_conf_list + depart_conf_list, approach_jawwidth_list + depart_jawwidth_list
        return [], []

    def gen_approach_motion(self,
                            component_name,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            start_conf=None,
                            approach_direction=None,
                            approach_distance=.1,
                            approach_jawwidth=.05,
                            granularity=.03,
                            obstacle_list=[],
                            object_list=[],
                            seed_jnt_values=None,
                            toggle_end_grasp=False,
                            end_jawwidth=.0,
                            max_time=300):
        """
        生成接近运动
        :param component_name: 组件名称
        :param goal_tcp_pos: 目标 TCP 位置
        :param goal_tcp_rotmat: 目标 TCP 旋转矩阵
        :param start_conf: 起始关节配置
        :param approach_direction: 接近方向,默认为旋转矩阵的 z 轴
        :param approach_distance: 接近距离
        :param approach_jawwidth: 接近时的夹爪宽度
        :param granularity: 步长
        :param obstacle_list: 障碍物列表
        :param object_list: 目标对象列表
        :param seed_jnt_values: 初始关节值
        :param toggle_end_grasp: 是否在结束时添加不同夹爪宽度的配置
        :param end_jawwidth: 结束时的夹爪宽度
        :return: 接近配置列表、夹爪宽度列表
        """
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        if approach_direction is None:
            # 如果未提供接近方向,默认为旋转矩阵的 z 轴
            approach_direction = goal_tcp_rotmat[:, 2]

        conf_list, jawwidth_list = self.gen_approach_linear(component_name,
                                                            goal_tcp_pos,
                                                            goal_tcp_rotmat,
                                                            approach_direction,
                                                            approach_distance,
                                                            approach_jawwidth,
                                                            granularity,
                                                            [],
                                                            seed_jnt_values,
                                                            toggle_end_grasp,
                                                            end_jawwidth)
        start2approach_conf_list = []
        start2approach_jawwidth_list = []

        if conf_list is not None:
            print("conf_list!", conf_list)
            print("len(conf_list):", len(conf_list))
        if conf_list is None:
            print("ADPlanner: 无法生成接近线性路径!")
            return None, None

        if start_conf is not None:
            # 使用 RRT 规划从起始配置到接近配置的路径
            start2approach_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                              start_conf=start_conf,
                                                              goal_conf=conf_list[0],
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=.2,
                                                              max_time=max_time)
            if start2approach_conf_list is not None:
                print("start2approach_conf_list:", start2approach_conf_list)
                print("len(start2approach_conf_list):", len(start2approach_conf_list))
            if start2approach_conf_list is None:
                print("ADPlanner: 无法规划接近运动!")
                return None, None
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)
        return start2approach_conf_list + conf_list, start2approach_jawwidth_list + jawwidth_list

    def gen_depart_motion(self,
                          component_name,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          end_conf=None,
                          depart_direction=None,
                          depart_distance=.1,
                          depart_jawwidth=.05,
                          granularity=.03,
                          obstacle_list=[],
                          object_list=[],
                          seed_jnt_values=None,
                          toggle_begin_grasp=False,
                          begin_jawwidth=.0):
        """
        生成离开运动

        :param component_name: 组件名称
        :param start_tcp_pos: 起始 TCP 位置
        :param start_tcp_rotmat: 起始 TCP 旋转矩阵
        :param end_conf: 结束关节配置
        :param depart_direction: 离开方向,默认为旋转矩阵的负 z 轴
        :param depart_distance: 离开距离
        :param depart_jawwidth: 离开时的夹爪宽度
        :param granularity: 步长
        :param obstacle_list: 障碍物列表
        :param object_list: 目标对象列表
        :param seed_jnt_values: 初始关节值
        :param toggle_begin_grasp: 是否在开始时添加不同夹爪宽度的配置
        :param begin_jawwidth: 开始时的夹爪宽度
        :return: 离开配置列表、夹爪宽度列表
        """
        if seed_jnt_values is None:
            seed_jnt_values = end_conf

        if depart_direction is None:
            # 如果未提供离开方向,默认为旋转矩阵的负 z 轴
            depart_direction = -start_tcp_rotmat[:, 2]

        conf_list, jawwidth_list = self.gen_depart_linear(component_name,
                                                          start_tcp_pos,
                                                          start_tcp_rotmat,
                                                          depart_direction,
                                                          depart_distance,
                                                          depart_jawwidth,
                                                          granularity,
                                                          obstacle_list,
                                                          seed_jnt_values,
                                                          toggle_begin_grasp,
                                                          begin_jawwidth)
        if conf_list is None:
            print("ADPlanner: 无法生成离开线性路径！")
            return None, None
        if end_conf is not None:
            # 使用 RRT 规划从离开配置到目标配置的路径
            depart2goal_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                           start_conf=conf_list[-1],
                                                           goal_conf=end_conf,
                                                           obstacle_list=obstacle_list + object_list,
                                                           ext_dist=.05,
                                                           max_time=300)
            if depart2goal_conf_list is None:
                print("ADPlanner: 无法规划离开运动!")
                return None, None
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
        else:
            depart2goal_conf_list = []
            depart2goal_jawwidth_list = []
        return conf_list + depart2goal_conf_list, jawwidth_list + depart2goal_jawwidth_list

    def gen_approach_and_depart_motion(self,
                                       component_name,
                                       goal_tcp_pos,
                                       goal_tcp_rotmat,
                                       start_conf=None,
                                       goal_conf=None,
                                       approach_direction=None,  # np.array([0, 0, -1])
                                       approach_distance=.1,
                                       approach_jawwidth=.05,
                                       depart_direction=None,  # np.array([0, 0, 1])
                                       depart_distance=.1,
                                       depart_jawwidth=0,
                                       granularity=.03,
                                       obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                                       object_list=[],  # target objects, will be checked by rrt, but not by linear
                                       seed_jnt_values=None):
        """
        生成接近和离开运动

        如果 seed_jnt_values 和 end_conf 都为 None,则退化为 gen_ad_primitive

        :param component_name: 组件名称
        :param goal_tcp_pos: 目标 TCP 位置
        :param goal_tcp_rotmat: 目标 TCP 旋转矩阵
        :param start_conf: 起始关节配置
        :param goal_conf: 目标关节配置
        :param approach_direction: 接近方向
        :param approach_distance: 接近距离
        :param approach_jawwidth: 接近时的夹爪宽度
        :param depart_direction: 离开方向
        :param depart_distance: 离开距离
        :param depart_jawwidth: 离开时的夹爪宽度
        :param granularity: 步长
        :param seed_jnt_values: 初始关节值
        :param obstacle_list: 障碍物列表
        :param object_list: 目标对象列表
        :return: 关节配置列表和夹爪宽度列表
        author: weiwei
        date: 20210113, 20210125
        """
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        if approach_direction is None:
            # 如果未提供接近方向,默认为旋转矩阵的 z 轴
            approach_direction = goal_tcp_rotmat[:, 2]
        if depart_direction is None:
            # 如果未提供离开方向,默认为旋转矩阵的负 z 轴
            approach_direction = -goal_tcp_rotmat[:, 2]

        ad_conf_list, ad_jawwidth_list = self.gen_approach_and_depart_linear(component_name,
                                                                             goal_tcp_pos,
                                                                             goal_tcp_rotmat,
                                                                             approach_direction,
                                                                             approach_distance,
                                                                             approach_jawwidth,
                                                                             depart_direction,
                                                                             depart_distance,
                                                                             depart_jawwidth,
                                                                             granularity,
                                                                             obstacle_list,
                                                                             seed_jnt_values)
        if ad_conf_list is None or len(ad_conf_list) == 0:
            print("ADPlanner: 无法生成接近和离开线性路径！")
            return None, None

        if start_conf is not None:
            # 使用 RRT 规划从起始配置到接近配置的路径
            start2approach_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                              start_conf=start_conf,
                                                              goal_conf=ad_conf_list[0],
                                                              obstacle_list=obstacle_list,
                                                              object_list=object_list,
                                                              ext_dist=.05,
                                                              max_time=300)
            if start2approach_conf_list is None:
                print("ADPlanner: 无法规划接近运动！")
                return None, None
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)

        if goal_conf is not None:
            # 使用 RRT 规划从离开配置到目标配置的路径
            depart2goal_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                           start_conf=ad_conf_list[-1],
                                                           goal_conf=goal_conf,
                                                           obstacle_list=obstacle_list,
                                                           object_list=object_list,
                                                           ext_dist=.05,
                                                           max_time=300)
            if depart2goal_conf_list is None:
                print("ADPlanner: 无法规划离开运动！")
                return None, None
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
        return start2approach_conf_list + ad_conf_list + depart2goal_conf_list, \
               start2approach_jawwidth_list + ad_jawwidth_list + depart2goal_jawwidth_list

    def gen_depart_and_approach_linear(self):
        # 生成离开和接近的线性路径
        pass

    def gen_depart_and_approach_motion(self):
        # 生成离开和接近的运动
        pass


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.robots.gofa5.gofa5_dh76 as gofa5
    import robot_sim.end_effectors.gripper.dh76.dh76 as dh

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    gofa_instance = gofa5.GOFA5(enable_cc=True)
    gofa_instance.gen_meshmodel().attach_to(base)

    yumi_instance = ym.Yumi(enable_cc=True)
    # yumi_instance.gen_meshmodel().attach_to(base)
    # manipulator_name = 'rgt_arm'
    # hnd_name = 'rgt_hnd'

    component_name = 'arm'
    hnd_name = 'hnd'
    gripper_s = dh.Dh76(fingertip_type='r_76')

    goal_pos = np.array([.4, .4, .3])
    goal_rotmat = np.eye(3)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    # goal_rotmat = rm.rotmat_from_axangle([0, 0, 0],math.ni)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    # gofa_instance.gen_meshmodel().attach_to(base)
    # base.run()

    # adp = ADPlanner(yumi_instance)
    adp = ADPlanner(gofa_instance)
    print("adp", adp)
    tic = time.time()

    # conf_list, jawwidth_list = adp.gen_ad_primitive(hnd_name,
    #                                                 goal_pos,
    #                                                 goal_rotmat,
    #                                                 approach_direction=np.array([0, 0, -1]),
    #                                                 approach_distance=.1,
    #                                                 depart_direction=np.array([0, 1, 0]),
    #                                                 depart_distance=.0,
    #                                                 depart_jawwidth=0)

    conf_list, jawwidth_list = adp.gen_approach_and_depart_motion(component_name,
                                                                  goal_pos,
                                                                  goal_rotmat,
                                                                  start_conf=gofa_instance.get_jnt_values(
                                                                      component_name),
                                                                  goal_conf=None,
                                                                  approach_direction=np.array([0, 0, -1]),
                                                                  approach_distance=.1,
                                                                  depart_direction=np.array([0, -1, 0]),
                                                                  depart_distance=.0,
                                                                  depart_jawwidth=0)

    # conf_list, jawwidth_list = adp.gen_approach_motion(hnd_name,
    #                                                    goal_pos,
    #                                                    goal_rotmat,
    #                                                    seed_jnt_values=gofa_instance.get_jnt_values(hnd_name),
    #                                                    approach_direction=np.array([0, 0, -1]),
    #                                                    approach_distance=.1)

    # conf_list, jawwidth_list = adp.gen_depart_motion(hnd_name,
    #                                                  goal_pos,
    #                                                  goal_rotmat,
    #                                                  end_conf=gofa_instance.get_jnt_values(hnd_name),
    #                                                  depart_direction=np.array([0, 0, 1]),
    #                                                  depart_distance=.1)

    print("conf_list:", conf_list)

    toc = time.time()
    print(toc - tic)

    if conf_list is None:
        print("Failed to generate motion path.")
    else:
        for i, conf_value in enumerate(conf_list):
            gofa_instance.fk(component_name, conf_value)
            gofa_instance.jaw_to(hnd_name, jawwidth_list[i])
            yumi_meshmodel = gofa_instance.gen_meshmodel()
            yumi_meshmodel.attach_to(base)
            gofa_instance.show_cdprimit()
    base.run()
