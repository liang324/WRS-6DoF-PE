import math
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm


class IncrementalNIK(object):
    def __init__(self, robot_s):
        self.robot_s = robot_s

    def gen_linear_motion(self,
                          component_name,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          goal_tcp_pos,
                          goal_tcp_rotmat,
                          obstacle_list=[],
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_debug=False):
        """
        生成线性运动路径

        :param component_name: 机器人组件的名称
        :param start_tcp_pos: 起始TCP位置
        :param start_tcp_rotmat: 起始TCP旋转矩阵
        :param goal_tcp_pos: 目标TCP位置
        :param goal_tcp_rotmat: 目标TCP旋转矩阵
        :param obstacle_list: 障碍物列表
        :param granularity: 插值的粒度
        :param seed_jnt_values: 初始关节角度值
        :param toggle_debug: 是否开启调试模式
        :return: 关节角度值列表,如果无法生成路径则返回None

        author: weiwei
        date: 20210125
        """
        # 备份当前关节角度值
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # 生成位置和旋转矩阵的插值列表
        pos_list, rotmat_list = rm.interplate_pos_rotmat(start_tcp_pos,
                                                         start_tcp_rotmat,
                                                         goal_tcp_pos,
                                                         goal_tcp_rotmat,
                                                         granularity=granularity)
        jnt_values_list = []
        # 如果没有提供初始关节角度值,则使用备份的值
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk
        # 遍历插值列表,计算每个插值点的逆运动学解
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.robot_s.ik(component_name, pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("在生成线性运动时,IK不可解！ IK not solvable in gen_linear_motion!")
                # 如果逆运动学解不可解,恢复原始关节角度值
                self.robot_s.fk(component_name, jnt_values_bk)
                return None
            else:
                # 更新机器人姿态
                self.robot_s.fk(component_name, jnt_values)
                # 检查是否与障碍物碰撞
                cd_result, ct_points = self.robot_s.is_collided(obstacle_list, toggle_contact_points=True)
                if cd_result:
                    if toggle_debug:
                        for ct_pnt in ct_points:
                            gm.gen_sphere(ct_pnt).attach_to(base)
                    print("在生成线性运动时,中间姿态发生碰撞！")
                    self.robot_s.fk(component_name, jnt_values_bk)
                    return None
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.robot_s.fk(component_name, jnt_values_bk)
        return jnt_values_list

    def gen_rel_linear_motion(self,
                              component_name,
                              goal_tcp_pos,
                              goal_tcp_rotmat,
                              direction,
                              distance,
                              obstacle_list=[],
                              granularity=0.03,
                              seed_jnt_values=None,
                              type='sink',
                              toggle_debug=False):
        """
        生成相对线性运动路径

        :param component_name: 机器人组件的名称
        :param goal_tcp_pos: 目标 TCP(工具中心点)的位置
        :param goal_tcp_rotmat: 目标 TCP 的旋转矩阵
        :param direction: 运动的方向向量
        :param distance: 运动的距离
        :param obstacle_list: 障碍物列表
        :param granularity: 运动的粒度,决定路径的细分程度
        :param seed_jnt_values: 初始关节值,用于逆向运动学求解的起始点
        :param type: 运动类型,'sink' 表示从目标位置向后运动,'source' 表示从目标位置向前运动
        :param toggle_debug: 是否开启调试模式
        :return: 生成的运动路径
        author: weiwei
        date: 20210114
        """
        if type == 'sink':
            # 计算起始位置为目标位置减去方向向量乘以距离
            start_tcp_pos = goal_tcp_pos - rm.unit_vector(direction) * distance
            start_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(component_name,
                                          start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        elif type == 'source':
            # 起始位置为目标位置,目标位置为目标位置加上方向向量乘以距离
            start_tcp_pos = goal_tcp_pos
            start_tcp_rotmat = goal_tcp_rotmat
            goal_tcp_pos = goal_tcp_pos + direction * distance
            goal_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(component_name,
                                          start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        else:
            raise ValueError("类型必须是 'sink' 或 'source'！ Type must be sink or source!")

    def gen_rel_linear_motion_with_given_conf(self,
                                              component_name,
                                              goal_jnt_values,
                                              direction,
                                              distance,
                                              obstacle_list=[],
                                              granularity=0.03,
                                              seed_jnt_values=None,
                                              type='sink',
                                              toggle_debug=False):
        """
        根据给定的关节配置生成相对线性运动路径

        :param component_name: 机器人组件的名称
        :param goal_jnt_values: 目标关节值
        :param direction: 运动的方向向量
        :param distance: 运动的距离
        :param obstacle_list: 障碍物列表
        :param granularity: 运动的粒度,决定路径的细分程度
        :param seed_jnt_values: 初始关节值,用于逆向运动学求解的起始点
        :param type: 运动类型,'sink' 表示从目标位置向后运动,'source' 表示从目标位置向前运动
        :param toggle_debug: 是否开启调试模式
        :return: 生成的运动路径
        author: weiwei
        date: 20210114
        """
        # 将关节配置转换为 TCP 位置和旋转矩阵
        goal_tcp_pos, goal_tcp_rotmat = self.robot_s.cvt_conf_to_tcp(component_name, goal_jnt_values)
        if type == 'sink':
            # 计算起始位置为目标位置减去方向向量乘以距离
            start_tcp_pos = goal_tcp_pos - rm.unit_vector(direction) * distance
            start_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(component_name,
                                          start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        elif type == 'source':
            # 起始位置为目标位置,目标位置为目标位置加上方向向量乘以距离
            start_tcp_pos = goal_tcp_pos
            start_tcp_rotmat = goal_tcp_rotmat
            goal_tcp_pos = goal_tcp_pos + direction * distance
            goal_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(component_name,
                                          start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        else:
            raise ValueError("类型必须是 'sink' 或 'source'！ Type must be sink or source!")

    def get_rotational_motion(self,
                              component_name,
                              start_tcp_pos,
                              start_tcp_rotmat,
                              goal_tcp_pos,
                              goal_tcp_rotmat,
                              obstacle_list=[],
                              rot_center=np.zeros(3),
                              rot_axis=np.array([1, 0, 0]),
                              granularity=0.03,
                              seed_jnt_values=None):
        # TODO
        pass

    def gen_circular_motion(self,
                            component_name,
                            circle_center_pos,
                            circle_ax,
                            start_tcp_rotmat,
                            end_tcp_rotmat,
                            radius=.02,
                            obstacle_list=[],
                            granularity=0.03,
                            seed_jnt_values=None,
                            toggle_tcp_list=False,
                            toggle_debug=False):
        """
        生成圆周运动路径

        :param component_name: 机器人组件的名称
        :param circle_center_pos: 圆心的位置
        :param circle_ax: 圆的轴向
        :param start_tcp_rotmat: 起始 TCP 旋转矩阵
        :param end_tcp_rotmat: 结束 TCP 旋转矩阵
        :param radius: 圆的半径
        :param obstacle_list: 障碍物列表
        :param granularity: 运动的粒度,决定路径的细分程度
        :param seed_jnt_values: 初始关节值,用于逆向运动学求解的起始点
        :param toggle_tcp_list: 是否返回 TCP 列表
        :param toggle_debug: 是否开启调试模式
        :return: 生成的关节值列表,或者关节值和 TCP 列表
        author: weiwei
        date: 20210501
        """
        # 备份当前的关节值
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # 计算圆周上的位置和旋转矩阵列表
        pos_list, rotmat_list = rm.interplate_pos_rotmat_around_circle(circle_center_pos,
                                                                       circle_ax,
                                                                       radius,
                                                                       start_tcp_rotmat,
                                                                       end_tcp_rotmat,
                                                                       granularity=granularity)
        # for (pos, rotmat) in zip(pos_list, rotmat_list):
        #     gm.gen_frame(pos, rotmat).attach_to(base)
        # base.run()

        # 初始化关节值列表
        jnt_values_list = []
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk

        # 遍历位置和旋转矩阵列表,计算每个点的逆向运动学
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.robot_s.ik(component_name, pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("在 gen_circular_motion 中无法求解 IK！IK not solvable in gen_circular_motion!")
                self.robot_s.fk(component_name, jnt_values_bk)
                return []
            else:
                self.robot_s.fk(component_name, jnt_values)
                # 检查是否与障碍物碰撞
                cd_result, ct_points = self.robot_s.is_collided(obstacle_list, toggle_contact_points=True)
                if cd_result:
                    if toggle_debug:
                        for ct_pnt in ct_points:
                            gm.gen_sphere(ct_pnt).attach_to(base)
                    print("在 gen_circular_motion 中,中间姿态发生碰撞！Intermediate pose collided in gen_linear_motion!")
                    self.robot_s.fk(component_name, jnt_values_bk)
                    return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.robot_s.fk(component_name, jnt_values_bk)
        if toggle_tcp_list:
            return jnt_values_list, [[pos_list[i], rotmat_list[i]] for i in range(len(pos_list))]
        else:
            return jnt_values_list


if __name__ == '__main__':
    import time
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.robots.gofa5.gofa5_dh76 as gofa5
    import robot_sim.end_effectors.gripper.dh76.dh76 as dh

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)
    # yumi_instance = ym.Yumi(enable_cc=True)
    robot_s = gofa5.GOFA5(enable_cc=True)

    robot_s.gen_meshmodel().attach_to(base)
    component_name = 'arm'
    start_pos = np.array([.5, -.3, .3])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_pos = np.array([.55, .3, .5])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)

    inik = IncrementalNIK(robot_s)
    tic = time.time()
    jnt_values_list = inik.gen_linear_motion(component_name, start_tcp_pos=start_pos, start_tcp_rotmat=start_rotmat,
                                             goal_tcp_pos=goal_pos, goal_tcp_rotmat=goal_rotmat)
    toc = time.time()
    print(toc - tic)

    # for jnt_values in jnt_values_list:
    #     yumi_instance.fk(component_name, jnt_values)
    #     yumi_meshmodel = yumi_instance.gen_meshmodel()
    #     yumi_meshmodel.attach_to(base)

    # 检查返回值是否有效
    if jnt_values_list:
        for jnt_values in jnt_values_list:
            robot_s.fk(component_name, jnt_values)
            yumi_meshmodel = robot_s.gen_meshmodel()
            yumi_meshmodel.attach_to(base)
    else:
        print("运动生成失败,无法迭代关节值列表")

    base.run()
