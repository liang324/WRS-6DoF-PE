# 定义一个名为 PickPlacePlanner 的类,继承自 adp.ADPlanner,用于生成机器人抓取和放置的运动轨迹
import math
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
import basis.robot_math as rm
import basis.data_adapter as da
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp
import time


class PickPlacePlanner(adp.ADPlanner):
    def __init__(self, robot_s):
        """
        :param objects:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(robot_s)

    def gen_object_motion(self, component_name, conf_list, obj_pos, obj_rotmat, type='absolute'):
        """
        生成物体的运动轨迹
        :param component_name: 组件名称,用于获取关节配置
        :param conf_list: 关节配置列表,表示组件在不同状态下的位置
        :param obj_pos: 物体的初始位置
        :param obj_rotmat: 物体的初始旋转矩阵
        :param type: 'absolute' 或 'relative',表示位置类型
        :return: 物体运动的齐次变换矩阵列表
        author: weiwei
        date: 20210125
        """
        objpose_list = []  # 初始化物体位置列表
        if type == 'absolute':  # 如果类型为绝对位置
            for _ in conf_list:  # 遍历每个关节配置
                # 生成物体的齐次变换矩阵并添加到列表中
                objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_rotmat))

        elif type == 'relative':  # 如果类型为相对位置
            jnt_values_bk = self.robot_s.get_jnt_values(component_name)  # 保存当前关节配置
            for conf in conf_list:  # 遍历每个关节配置
                self.robot_s.fk(component_name, conf)  # 进行正向运动学计算,更新组件位置
                # 将物体的位置和旋转矩阵转换为全局坐标系
                gl_obj_pos, gl_obj_rotmat = self.robot_s.cvt_loc_tcp_to_gl(component_name, obj_pos, obj_rotmat)
                # 生成物体的齐次变换矩阵并添加到列表中
                objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
            self.robot_s.fk(component_name, jnt_values_bk)  # 恢复到之前的关节配置

        else:  # 如果类型不符合要求
            raise ValueError('Type must be absolute or relative!')  # 抛出异常
        return objpose_list  # 返回物体运动的齐次变换矩阵列表

    def find_common_graspids(self,
                             hand_name,  # TODO: 手部名称,表示一个组件可能有多个手
                             grasp_info_list,  # 抓取信息列表,格式为 [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
                             goal_homomat_list,  # 目标位姿的齐次变换矩阵列表
                             obstacle_list=[],  # 障碍物列表,默认为空
                             toggle_debug=False):  # 调试模式开关,默认为关闭
        """
        找到无碰撞且逆运动学可行的通用抓取 ID
        查找可行的抓取 ID,确保抓取在目标位置时不会发生碰撞,并且能够通过逆运动学(IK)求解
        统计碰撞和 IK 失败的抓取次数,并在调试模式下输出相关信息
        :param hand_name: 一个组件可能有多个手
        :param grasp_info_list: 抓取信息列表
        :param goal_homomat_list: 目标齐次变换矩阵列表
        :param obstacle_list: 障碍物列表
        :return: [final_available_graspids, intermediate_available_graspids]
        """
        # 初始化相关变量
        hnd_instance = self.robot_s.hnd_dict[hand_name]  # 获取手部实例
        previously_available_graspids = range(len(grasp_info_list))  # 初始化可用抓取 ID 列表
        intermediate_available_graspids = []  # 存储每个目标位置的中间可用抓取 ID 列表
        hndcollided_grasps_num = 0  # 记录手部碰撞的抓取次数
        ikfailed_grasps_num = 0  # 记录逆运动学失败的抓取次数
        rbtcollided_grasps_num = 0  # 记录机器人碰撞的抓取次数
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)  # 保存当前手部的关节配置,以便后续恢复

        # 遍历每个目标位置
        for goalid, goal_homomat in enumerate(goal_homomat_list):
            goal_pos = goal_homomat[:3, 3]  # 提取目标位置
            goal_rotmat = goal_homomat[:3, :3]  # 提取目标旋转矩阵
            graspid_and_graspinfo_list = zip(previously_available_graspids,  # 将可用抓取 ID 和抓取信息配对
                                             [grasp_info_list[i] for i in previously_available_graspids])
            previously_available_graspids = []  # 重置可用抓取 ID 列表

            for graspid, grasp_info in graspid_and_graspinfo_list:
                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info  # 解包抓取信息
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(jaw_center_pos)  # 计算目标夹爪中心位置
                goal_jaw_center_rotmat = goal_rotmat.dot(jaw_center_rotmat)  # 计算目标夹爪旋转矩阵
                hnd_instance.grip_at_with_jcpose(goal_jaw_center_pos, goal_jaw_center_rotmat, jaw_width)  # 控制手部抓取

                # 检查手部是否与障碍物发生碰撞
                if not hnd_instance.is_mesh_collided(obstacle_list):  # 如果没有碰撞
                    jnt_values = self.robot_s.ik(hand_name, goal_jaw_center_pos, goal_jaw_center_rotmat)  # 进行逆运动学求解
                    if jnt_values is not None:  # 如果逆运动学求解成功
                        if toggle_debug:  # 如果调试模式开启
                            hnd_tmp = hnd_instance.copy()  # 复制手部实例
                            hnd_tmp.gen_meshmodel(rgba=[0, 1, 0, .2]).attach_to(base)  # 可视化手部模型
                        self.robot_s.fk(hand_name, jnt_values)  # 进行正向运动学

                        is_rbt_collided = self.robot_s.is_collided(obstacle_list)  # 检查机器人是否与障碍物发生碰撞
                        is_obj_collided = False  # 假设物体没有碰撞

                        # 检查手部、机器人和物体是否发生碰撞
                        if (not is_rbt_collided) and (not is_obj_collided):  # 如果都没有碰撞
                            if toggle_debug:  # 如果调试模式开启
                                self.robot_s.gen_meshmodel(rgba=[0, 1, 0, .5]).attach_to(base)  # 可视化机器人模型
                            previously_available_graspids.append(graspid)  # 添加可用抓取 ID

                        elif (not is_obj_collided):  # 如果物体没有碰撞,但机器人发生了碰撞
                            rbtcollided_grasps_num += 1  # 增加机器人碰撞计数
                            if toggle_debug:  # 如果调试模式开启
                                self.robot_s.gen_meshmodel(rgba=[1, 0, 1, .5]).attach_to(base)  # 可视化碰撞状态

                    else:  # 如果逆运动学求解失败
                        ikfailed_grasps_num += 1  # 增加逆运动学失败计数
                        if toggle_debug:  # 如果调试模式开启
                            hnd_tmp = hnd_instance.copy()  # 复制手部实例
                            hnd_tmp.gen_meshmodel(rgba=[1, .6, 0, .2]).attach_to(base)  # 可视化失败状态
                else:  # 如果手部发生碰撞
                    hndcollided_grasps_num += 1  # 增加手部碰撞计数
                    if toggle_debug:  # 如果调试模式开启
                        hnd_tmp = hnd_instance.copy()  # 复制手部实例
                        hnd_tmp.gen_meshmodel(rgba=[1, 0, 1, .2]).attach_to(base)  # 可视化碰撞状态

            intermediate_available_graspids.append(previously_available_graspids.copy())  # 保存当前目标位置的可用抓取 ID 列表

            # 输出每个目标位置的碰撞、IK失败和机器人碰撞的次数
            print('-----start-----')
            print('Number of hndcollided_grasps_num collided grasps at goal-' + str(goalid) + ': ',
                  hndcollided_grasps_num)  # 输出手部碰撞次数
            print('Number of ikfailed_grasps_num failed IK at goal-' + str(goalid) + ': ',
                  ikfailed_grasps_num)  # 输出逆运动学失败次数
            print('Number of rbtcollided_grasps_num collided robots at goal-' + str(goalid) + ': ',
                  rbtcollided_grasps_num)  # 输出机器人碰撞次数
            print('------end------')

        final_available_graspids = previously_available_graspids  # 最终可用抓取 ID 列表
        self.robot_s.fk(hand_name, jnt_values_bk)  # 恢复手部的关节配置

        # 最终返回两个抓取 ID 列表
        return final_available_graspids, intermediate_available_graspids  # 返回最终和中间可用抓取 ID 列表

    def gen_holding_rel_linear(self):
        pass

    def gen_holding_linear(self):
        pass

    def gen_holding_moveto(self,
                           hand_name,
                           objcm,
                           grasp_info,
                           obj_pose_list,
                           depart_direction_list,
                           depart_distance_list,
                           approach_direction_list,
                           approach_distance_list,
                           ad_granularity=.007,
                           use_rrt=True,
                           obstacle_list=[],
                           seed_jnt_values=None):
        """
        保持和移动一个对象到多个姿势

        :param hand_name: 手的名称
        :param grasp_info: 抓取信息
        :param obj_pose_list: 目标物体位置列表
        :param depart_direction_list: 离开方向列表,最后一个元素将被忽略
        :param depart_distance_list: 离开距离列表,最后一个元素将被忽略
        :param approach_direction_list: 接近方向列表,第一个元素将被忽略
        :param approach_distance_list: 接近距离列表,第一个元素将被忽略
        :param ad_granularity: 粒度
        :param obstacle_list: 障碍物列表
        :param seed_jnt_values: 初始关节配置
        :return: 关节配置列表、夹爪宽度列表和物体位置列表
        """
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)  # 获取当前关节配置并备份
        jawwidth_bk = self.robot_s.get_jawwidth(hand_name)  # 获取当前夹爪宽度并备份

        # 初始化最终的配置列表、夹爪宽度列表和物体位置列表
        conf_list = []
        jawwidthlist = []
        objpose_list = []

        # 获取抓取信息
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        first_obj_pos = obj_pose_list[0][:3, 3]  # 获取第一个目标物体的位置
        first_obj_rotmat = obj_pose_list[0][:3, :3]  # 获取第一个目标物体的旋转矩阵

        # # 计算第一个抓取位置的夹爪中心位置和旋转矩阵
        # first_jaw_center_pos = first_obj_rotmat.dot(jaw_center_pos) + first_obj_pos
        # first_jaw_center_rotmat = first_obj_rotmat.dot(jaw_center_rotmat)
        # print("seed_jnt_values", seed_jnt_values)
        # # 计算第一个抓取位置的逆运动学
        # first_conf = self.robot_s.ik(hand_name,
        #                              first_jaw_center_pos,
        #                              first_jaw_center_rotmat,
        #                              seed_jnt_values=seed_jnt_values)
        # print("first_conf", first_conf)
        # if first_conf is None:
        #     print("Cannot solve the ik at the first grasping pose!")
        #     return None, None, None
        self.robot_s.fk(component_name=hand_name, jnt_values=seed_jnt_values)

        # 设置物体的初始位置,抓住物体并移动到目标物体位置
        objcm_copy = objcm.copy()  # 复制物体模型
        objcm_copy.set_pos(first_obj_pos)  # 设置物体的初始位置
        objcm_copy.set_rotmat(first_obj_rotmat)  # 设置物体的初始旋转矩阵
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hand_name, objcm_copy, jaw_width)  # 抓住物体
        seed_conf = seed_jnt_values  # 将初始配置设置为种子配置

        for i in range(len(obj_pose_list) - 1):  # 遍历目标物体位置列表
            # 获取起始和目标物体的位置和旋转矩阵
            start_obj_pos = obj_pose_list[i][:3, 3]
            start_obj_rotmat = obj_pose_list[i][:3, :3]

            goal_obj_pos = obj_pose_list[i + 1][:3, 3]
            goal_obj_rotmat = obj_pose_list[i + 1][:3, :3]

            # 计算抓取位置
            start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
            start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)

            goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
            goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)

            # 获取离开方向和距离
            depart_direction = depart_direction_list[i]
            if depart_direction is None:  # 如果没有指定方向,使用默认方向
                depart_direction = -start_jaw_center_rotmat[:, 2]
            depart_distance = depart_distance_list[i]
            if depart_distance is None:  # 如果没有指定距离,默认为0
                depart_distance = 0

            # 获取接近方向和距离
            approach_direction = approach_direction_list[i + 1]
            if approach_direction is None:  # 如果没有指定方向,使用目标抓取的方向
                approach_direction = goal_jaw_center_rotmat[:, 2]
            approach_distance = approach_distance_list[i + 1]
            if approach_distance is None:  # 如果没有指定距离,默认为0
                approach_distance = 0

            print("depart_direction", depart_direction)
            print("depart_distance", depart_distance)
            print("approach_direction", approach_direction)

            # 生成离开运动的关节配置
            conf_list_depart = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                    goal_tcp_pos=start_jaw_center_pos,
                                                                    goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                    direction=depart_direction,
                                                                    distance=depart_distance,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf,
                                                                    type='source')
            if conf_list_depart is not None:
                print("conf_list_depart", conf_list_depart)
            elif conf_list_depart is None:
                print(f"Cannot generate the linear part of the {i}th holding depart motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)  # 释放物体
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)  # 恢复手的位置
                return None, None, None

            # 生成夹爪宽度变化和物体运动
            jawwidthlist_depart = self.gen_jawwidth_motion(conf_list_depart, jaw_width)
            objpose_list_depart = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_depart,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            ###########################
            if use_rrt:
                # 生成接近运动的关节配置
                seed_conf = conf_list_depart[-1]  # 使用离开运动的最后一个配置作为种子
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is not None:
                    print("conf_list_approach", conf_list_approach)
                if conf_list_approach is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")  # 打印错误信息
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)  # 释放物体
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)  # 恢复手的位置
                    return None, None, None

                # 规划中间运动
                conf_list_middle = self.rrtc_planner.plan(component_name=hand_name,
                                                          start_conf=conf_list_depart[-1],
                                                          goal_conf=conf_list_approach[0],
                                                          obstacle_list=obstacle_list,
                                                          otherrobot_list=[],
                                                          ext_dist=.07,
                                                          max_iter=300)
                if conf_list_middle is not None:
                    print("conf_list_middle", conf_list_middle)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")  # 打印错误信息
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)  # 释放物体
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)  # 恢复手的位置
                    return None, None, None

            else:
                seed_conf = conf_list_depart[-1]  # 使用离开运动的最后一个配置作为种子
                self.robot_s.fk(component_name=hand_name, jnt_values=seed_conf)  # 更新手的位置
                mid_start_tcp_pos, mid_start_tcp_rotmat = self.robot_s.get_gl_tcp(hand_name)  # 获取当前TCP位置和旋转矩阵
                mid_goal_tcp_pos = goal_jaw_center_pos - approach_direction * approach_distance  # 计算中间目标位置
                mid_goal_tcp_rotmat = goal_jaw_center_rotmat  # 中间目标旋转矩阵

                # 生成中间运动的关节配置
                conf_list_middle = self.inik_slvr.gen_linear_motion(component_name=hand_name,
                                                                    start_tcp_pos=mid_start_tcp_pos,
                                                                    start_tcp_rotmat=mid_start_tcp_rotmat,
                                                                    goal_tcp_pos=mid_goal_tcp_pos,
                                                                    goal_tcp_rotmat=mid_goal_tcp_rotmat,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf)
                if conf_list_middle is None:  # 如果无法生成中间运动
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")  # 打印错误信息
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)  # 释放物体
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)  # 恢复手的位置
                    return None, None, None  # 返回空值

                # 生成接近运动的关节配置
                seed_conf = conf_list_middle[-1]  # 使用中间运动的最后一个配置作为种子
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is None:  # 如果无法生成接近运动
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")  # 打印错误信息
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)  # 释放物体
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)  # 恢复手的位置
                    return None, None, None  # 返回空值

            # 生成夹爪宽度变化和物体运动
            jawwidthlist_approach = self.gen_jawwidth_motion(conf_list_approach, jaw_width)
            objpose_list_approach = self.gen_object_motion(component_name=hand_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=rel_obj_pos,
                                                           obj_rotmat=rel_obj_rotmat,
                                                           type='relative')

            jawwidthlist_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_middle,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')

            # 合并所有运动的配置、夹爪宽度和物体位置
            conf_list = conf_list + conf_list_depart + conf_list_middle + conf_list_approach
            jawwidthlist = jawwidthlist + jawwidthlist_depart + jawwidthlist_middle + jawwidthlist_approach
            objpose_list = objpose_list + objpose_list_depart + objpose_list_middle + objpose_list_approach

            seed_conf = conf_list[-1]  # 更新种子配置为当前的最后一个配置

        # 释放物体并恢复手的位置
        self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
        self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
        return conf_list, jawwidthlist, objpose_list  # 返回所有生成的配置、夹爪宽度和物体位置

    def gen_pick_and_place_motion(self,
                                  hnd_name,  # 手的名称
                                  objcm,  # 物体的控制模型
                                  start_conf,  # 起始关节配置
                                  end_conf,  # 结束关节配置
                                  grasp_info_list,  # 抓取信息列表
                                  goal_homomat_list,  # 目标位姿的齐次变换矩阵列表
                                  approach_direction_list,  # 接近方向列表
                                  approach_distance_list,  # 接近距离列表
                                  depart_direction_list,  # 离开方向列表
                                  depart_distance_list,  # 离开距离列表
                                  approach_jawwidth=None,  # 接近时的夹爪宽度
                                  depart_jawwidth=None,  # 离开时的夹爪宽度
                                  ad_granularity=.007,  # 接近和离开的粒度
                                  use_rrt=True,  # 是否使用RRT进行路径规划
                                  obstacle_list=[],  # 障碍物列表
                                  grasp_obstacle_list=[],
                                  use_incremental=False):  # 是否使用增量方式
        """
        生成抓取和放置的运动轨迹

        :param hnd_name: 手的名称
        :param objcm: 物体的控制模型
        :param grasp_info_list: 抓取信息列表
        :param goal_homomat_list: 目标位姿的齐次变换矩阵列表
        :param start_conf: 起始关节配置
        :param end_conf: 结束关节配置
        :param approach_direction_list: 接近方向列表
        :param approach_distance_list: 接近距离列表
        :param depart_direction_list: 离开方向列表
        :param depart_distance_list: 离开距离列表
        :param approach_jawwidth: 接近时的夹爪宽度
        :param depart_jawwidth: 离开时的夹爪宽度
        :param ad_granularity: 粒度
        :param use_rrt: 是否使用 RRT
        :param obstacle_list: 障碍物列表
        :param use_incremental: 是否使用增量方式
        :return: 关节配置、夹爪宽度和物体位置列表
        author: weiwei
        date: 20191122, 20200105
        """
        # 如果没有指定接近时的夹爪宽度,则使用最大夹爪宽度
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
        # 如果没有指定离开时的夹爪宽度,则使用最大夹爪宽度
        if depart_jawwidth is None:
            depart_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]

        # 获取第一个目标位置和旋转矩阵
        first_goal_pos = goal_homomat_list[0][:3, 3]
        first_goal_rotmat = goal_homomat_list[0][:3, :3]

        # 获取最后一个目标位置和旋转矩阵
        last_goal_pos = goal_homomat_list[-1][:3, 3]
        last_goal_rotmat = goal_homomat_list[-1][:3, :3]

        # 根据是否使用增量方式来确定抓取ID列表
        if use_incremental:
            common_grasp_id_list = range(len(grasp_info_list))  # 使用所有抓取ID
        else:
            # 找到与目标位姿匹配的抓取ID
            common_grasp_id_list, _ = self.find_common_graspids(hnd_name,
                                                                grasp_info_list,
                                                                goal_homomat_list,
                                                                obstacle_list=grasp_obstacle_list)

        if len(common_grasp_id_list) is not None:
            print("找到与目标位姿匹配的抓取 IDcommon_grasp_id_list的长度", len(common_grasp_id_list))
        elif len(common_grasp_id_list) == 0:
            print("No common grasp id at the given goal homomats!")
            return None, None, None

        start_time = time.time()

        for grasp_id in common_grasp_id_list:
            grasp_info = grasp_info_list[grasp_id]
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

            # 计算接近时的夹爪中心位置和旋转矩阵
            first_jaw_center_pos = first_goal_rotmat.dot(jaw_center_pos) + first_goal_pos
            first_jaw_center_rotmat = first_goal_rotmat.dot(jaw_center_rotmat)

            # 将物体控制模型复制并设置为障碍物
            objcm_copy = objcm.copy()  # 复制物体控制模型
            objcm_copy.set_pos(first_goal_pos)  # 设置物体位置
            objcm_copy.set_rotmat(first_goal_rotmat)  # 设置物体旋转矩阵

            #################### 生成接近运动的关节配置和夹爪宽度
            conf_list_approach, jawwidthlist_approach = \
                self.gen_approach_motion(component_name=hnd_name,
                                         goal_tcp_pos=first_jaw_center_pos,
                                         goal_tcp_rotmat=first_jaw_center_rotmat,
                                         start_conf=start_conf,
                                         approach_direction=approach_direction_list[0],
                                         approach_distance=approach_distance_list[0],
                                         approach_jawwidth=approach_jawwidth,
                                         granularity=ad_granularity,
                                         obstacle_list=obstacle_list,
                                         # object_list=[objcm_copy],  # 将物体作为障碍物
                                         object_list=[],  # 将物体作为障碍物
                                         seed_jnt_values=start_conf)  # 使用起始配置作为种子
            if conf_list_approach is not None:
                print("conf_list_approach", conf_list_approach)
                print("len(conf_list_approach)", len(conf_list_approach))
                print("jawwidthlist_approach", jawwidthlist_approach)
            # 如果无法生成接近运动,打印错误信息并继续下一个抓取ID
            elif conf_list_approach is None:
                print("无法产生拾取动作！")
                continue
            # jnt_values_bk = self.robot_s.get_jnt_values(hnd_name)
            # print("jnt_values_bk", jnt_values_bk) # jnt_values_bk [0. 0. 0. 0. 0. 0.]

            #################### 生成抓取和移动的中间运动
            conf_list_middle, jawwidthlist_middle, objpose_list_middle = \
                self.gen_holding_moveto(hand_name=hnd_name,
                                        objcm=objcm,
                                        grasp_info=grasp_info,
                                        obj_pose_list=goal_homomat_list,
                                        depart_direction_list=depart_direction_list,
                                        approach_direction_list=approach_direction_list,
                                        depart_distance_list=depart_distance_list,
                                        approach_distance_list=approach_distance_list,
                                        ad_granularity=.003,
                                        use_rrt=use_rrt,
                                        obstacle_list=obstacle_list,
                                        seed_jnt_values=conf_list_approach[-1])  # 使用接近运动的最后配置作为种子

            if conf_list_middle is not None:
                print("conf_list_middle", conf_list_middle)
                print("jawwidthlist_middle", jawwidthlist_middle)
                print("objpose_list_middle", objpose_list_middle)
                # 如果无法生成中间运动,继续下一个抓取ID
            elif conf_list_middle is None or len(conf_list_middle) == 0:
                print("Cannot generate the middle motion!")
                continue

            # 计算离开时的夹爪中心位置和旋转矩阵
            last_jaw_center_pos = last_goal_rotmat.dot(jaw_center_pos) + last_goal_pos
            last_jaw_center_rotmat = last_goal_rotmat.dot(jaw_center_rotmat)

            # 将物体控制模型设置为障碍物
            objcm_copy.set_pos(last_goal_pos)  # 设置物体位置
            objcm_copy.set_rotmat(last_goal_rotmat)  # 设置物体旋转矩阵

            # 生成离开运动的关节配置和夹爪宽度
            conf_list_depart, jawwidthlist_depart = \
                self.gen_depart_motion(component_name=hnd_name,
                                       start_tcp_pos=last_jaw_center_pos,
                                       start_tcp_rotmat=last_jaw_center_rotmat,
                                       end_conf=end_conf,
                                       depart_direction=depart_direction_list[-1],
                                       depart_distance=depart_distance_list[-1],
                                       depart_jawwidth=depart_jawwidth,
                                       granularity=ad_granularity,
                                       obstacle_list=obstacle_list,
                                       # object_list=[objcm_copy],  # 将物体作为障碍物
                                       object_list=[],
                                       seed_jnt_values=conf_list_middle[-1])  # 使用中间运动的最后配置作为种子
            if conf_list_depart is not None:
                print("生成离开运动的关节配置和夹爪宽度 conf_list_depart", conf_list_depart)
                # 如果无法生成离开运动,打印错误信息并继续下一个抓取ID
            elif conf_list_depart is None:
                print("Cannot generate the release motion!")
                continue

            # 生成接近时的物体运动
            objpose_list_approach = self.gen_object_motion(component_name=hnd_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=first_goal_pos,
                                                           obj_rotmat=first_goal_rotmat,
                                                           type='absolute')  # 绝对位置

            # 生成离开时的物体运动
            objpose_list_depart = self.gen_object_motion(component_name=hnd_name,
                                                         conf_list=conf_list_depart,
                                                         obj_pos=last_goal_pos,
                                                         obj_rotmat=last_goal_rotmat,
                                                         type='absolute')  # 绝对位置

            # return [conf_list_approach, conf_list_middle, conf_list_depart], \
            #     [jawwidthlist_approach, jawwidthlist_middle, jawwidthlist_depart], \
            #     [objpose_list_approach, objpose_list_middle, objpose_list_depart]

            # 返回所有生成的运动轨迹、夹爪宽度和物体位置
            return conf_list_approach + conf_list_middle + conf_list_depart, \
                   jawwidthlist_approach + jawwidthlist_middle + jawwidthlist_depart, \
                   objpose_list_approach + objpose_list_middle + objpose_list_depart

        print("time.time() - start_time", time.time() - start_time)

        # 如果没有找到有效的抓取ID,返回空值
        return None, None, None


if __name__ == '__main__':
    import time
    import robot_sim.robots.yumi.yumi as ym
    import robot_sim.robots.gofa5.gofa5_dh76 as gofa5
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import grasping.annotation.utils as gutil
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.end_effectors.gripper.dh76.dh76 as dh

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    objcm = cm.CollisionModel('rack_10ml_new.stl')
    # robot_s = ym.Yumi(enable_cc=True)

    robot_s = gofa5.GOFA5(enable_cc=True)
    robot_s.gen_meshmodel().attach_to(base)
    manipulator_name = 'arm'
    hand_name = 'hnd'
    gripper_s = dh.Dh76(fingertip_type='r_76')

    start_conf = robot_s.get_jnt_values(manipulator_name)
    print(start_conf)

    goal_homomat_list = []

    for i in range(1):
        goal_pos = np.array([.4, .1, .3]) - np.array([i * .1, i * .1, 0])
        # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        goal_rotmat = np.eye(3)
        goal_homomat_list.append(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm = objcm.copy()
        tmp_objcm.set_rgba([1, 0, 0, .3])
        tmp_objcm.set_homomat(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm.attach_to(base)

    grasp_info_list = gutil.load_pickle_file(objcm_name='rack_10ml_new', file_name='dh76_grasps_rack_10ml_new.pickle')
    grasp_info = grasp_info_list[0]

    pp_planner = PickPlacePlanner(robot_s=robot_s)
    conf_list, jawwidth_list, objpose_list = \
        pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
                                             objcm=objcm,
                                             grasp_info_list=grasp_info_list,
                                             goal_homomat_list=goal_homomat_list,
                                             start_conf=robot_s.get_jnt_values(hand_name),
                                             end_conf=robot_s.get_jnt_values(hand_name),
                                             depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
                                             approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
                                             # depart_distance_list=[None] * len(goal_homomat_list),
                                             # approach_distance_list=[None] * len(goal_homomat_list),
                                             depart_distance_list=[.2] * len(goal_homomat_list),
                                             approach_distance_list=[.2] * len(goal_homomat_list),
                                             approach_jawwidth=None,
                                             depart_jawwidth=None,
                                             ad_granularity=.003,
                                             use_rrt=True,
                                             obstacle_list=[],
                                             use_incremental=False)
    # for grasp_info in grasp_info_list:
    #     conf_list, jawwidth_list, objpose_list = \
    #         pp_planner.gen_holding_moveto(hnd_name='rgt_hnd',
    #                                       objcm=objcm,
    #                                       grasp_info=grasp_info,
    #                                       obj_pose_list=goal_homomat_list,
    #                                       depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
    #                                       approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
    #                                       # depart_distance_list=[None] * len(goal_homomat_list),
    #                                       # approach_distance_list=[None] * len(goal_homomat_list),
    #                                       depart_distance_list=[.2] * len(goal_homomat_list),
    #                                       approach_distance_list=[.2] * len(goal_homomat_list),
    #                                       ad_granularity=.003,
    #                                       use_rrt=True,
    #                                       obstacle_list=[],
    #                                       seed_jnt_values=start_conf)
    #     print(robot_s.rgt_oih_infos, robot_s.lft_oih_infos)
    #     if conf_list is not None:
    #         break

    # animation
    robot_attached_list = []
    object_attached_list = []
    counter = [0]


    def update(robot_s,
               hand_name,  # 机器人的手的名称
               objcm,  # 物体的控制模型
               robot_path,  # 机器人运动路径的列表
               jawwidth_path,  # 机器人夹具的宽度列表
               obj_path,  # 物体运动路径的列表
               robot_attached_list,  # 当前附着在场景中的机器人模型列表
               object_attached_list,  # 当前附着在场景中的物体模型列表
               counter,  # 计数器,用于跟踪当前路径索引
               task):  # 任务对象,用于调度更新

        # 参数检查
        if robot_path is None or len(robot_path) == 0:
            print("Error: robot_path is None or empty!")
            return task.done

        if jawwidth_path is None or len(jawwidth_path) == 0:
            print("Error: jawwidth_path is None or empty!")
            return task.done

        if obj_path is None or len(obj_path) == 0:
            print("Error: obj_path is None or empty!")
            return task.done

        # 检查计数器是否超出机器人路径的长度
        if counter[0] >= len(robot_path):
            counter[0] = 0  # 如果超出,重置计数器为0

        # 如果有附着的机器人或物体
        if len(robot_attached_list) != 0:
            # 遍历并分离所有附着的机器人
            for robot_attached in robot_attached_list:
                robot_attached.detach()  # 从场景中分离机器人

            # 遍历并分离所有附着的物体
            for object_attached in object_attached_list:
                object_attached.detach()  # 从场景中分离物体

            # 清空附着列表
            robot_attached_list.clear()  # 清空机器人附着列表
            object_attached_list.clear()  # 清空物体附着列表

        # 获取当前路径中的机器人姿态
        pose = robot_path[counter[0]]  # 当前计数器索引对应的机器人姿态
        robot_s.fk(hand_name, pose)  # 执行正向运动学,更新机器人的姿态

        # 设置机器人的夹具宽度
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])  # 设置夹具宽度

        # 生成机器人的网格模型
        robot_meshmodel = robot_s.gen_meshmodel()  # 生成机器人模型的网格表示
        robot_meshmodel.attach_to(base)  # 将机器人模型附加到场景的基础上
        robot_attached_list.append(robot_meshmodel)  # 将机器人模型添加到附着列表中

        # 获取当前路径中的物体姿态
        obj_pose = obj_path[counter[0]]  # 当前计数器索引对应的物体姿态
        objb_copy = objcm.copy()  # 复制物体的控制模型
        objb_copy.set_homomat(obj_pose)  # 设置物体的变换矩阵
        objb_copy.attach_to(base)  # 将物体模型附加到场景的基础上
        object_attached_list.append(objb_copy)  # 将物体模型添加到附着列表中

        # 增加计数器以准备下次更新
        counter[0] += 1  # 增加计数器

        return task.again  # 返回任务以继续更新


    # 调度更新任务,每0.01秒调用一次update函数
    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot_s,  # 传递机器人控制对象
                                     hand_name,  # 传递手的名称
                                     objcm,  # 传递物体控制模型
                                     conf_list,  # 传递机器人配置列表
                                     jawwidth_list,  # 传递夹具宽度列表
                                     objpose_list,  # 传递物体姿态列表
                                     robot_attached_list,  # 传递附着的机器人列表
                                     object_attached_list,  # 传递附着的物体列表
                                     counter],  # 传递计数器
                          appendTask=True)  # 追加任务以确保更新

    base.run()
