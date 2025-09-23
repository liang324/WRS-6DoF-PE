import copy
import math
import time
import random
import numpy as np
import warnings as wns
from scipy.optimize import minimize
import basis.robot_math as rm


class FKOptBasedIK(object):
    """
    基于正向运动学和优化的逆运动学求解器

    通过优化关节角度,使机器人末端执行器达到目标位置和方向
    """

    def __init__(self, robot, component_name, obstacle_list=[], toggle_debug=False):
        """
        初始化 FKOptBasedIK 类的实例

        :param robot: 机器人模型
        :param component_name: 组件名称
        :param obstacle_list: 障碍物列表
        :param toggle_debug: 是否开启调试模式
        """
        self.rbt = robot
        self.jlc_name = component_name
        self.result = None
        self.seed_jnt_values = None
        self.tgt_pos = None
        self.tgt_rotmat = None
        self.cons = []  # 约束条件列表
        self.bnds = self._get_bnds(jlc_name=self.jlc_name)  # 关节角度范围
        self.jnts = []  # 优化过程中关节角度的记录
        self.jnt_diff = []  # 关节角度的变化量记录
        self.xangle_err = []  # X轴角度误差记录
        self.zangle_err = []  # Z轴角度误差记录
        self.x_err = []  # X轴位置误差
        self.y_err = []  # Y轴位置误差
        self.z_err = []  # Z轴位置误差
        self._xangle_limit = math.pi / 360  # X轴角度误差限制小于0.5度
        self._zangle_limit = math.pi / 6  # Z轴角度误差限制
        self._x_limit = 1e-6  # X轴位置误差限制
        self._y_limit = 1e-6  # Y轴位置误差限制
        self._z_limit = 1e-6  # Z轴位置误差限制
        self.obstacle_list = obstacle_list
        self.toggle_debug = toggle_debug

    def _get_bnds(self, jlc_name):
        """
        获取关节的范围,用于设置优化问题的边界条件

        :param jlc_name: 关节链名称
        :return: 关节范围列表
        """
        return self.rbt.get_jnt_ranges(jlc_name)

    def _constraint_zangle(self, jnt_values):
        """
        计算当前关节配置下,末端执行器的Z轴方向与目标方向之间的角度差

        :param jnt_values: 当前关节角度
        :return: Z轴角度误差
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rotmat = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        delta_angle = rm.angle_between_vectors(gl_tcp_rotmat[:, 2], self.tgt_rotmat[:, 2])
        self.zangle_err.append(delta_angle)
        return self._zangle_limit - delta_angle

    def _constraint_xangle(self, jnt_values):
        """
        计算当前关节配置下,末端执行器的X轴方向与目标方向之间的角度差

        :param jnt_values: 当前关节角度
        :return: X轴角度误差
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rotmat = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        delta_angle = rm.angle_between_vectors(gl_tcp_rotmat[:, 0], self.tgt_rotmat[:, 0])
        self.xangle_err.append(delta_angle)
        return self._xangle_limit - delta_angle

    def _constraint_x(self, jnt_values):
        """
        计算当前关节配置下,末端执行器的X轴位置与目标位置之间的误差

        :param jnt_values: 当前关节角度
        :return: X轴位置误差
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.x_err.append(x_err)
        return self._x_limit - x_err

    def _constraint_y(self, jnt_values):
        """
        计算当前关节配置下,末端执行器的Y轴位置与目标位置之间的误差.

        :param jnt_values: 当前关节角度
        :return: Y轴位置误差
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        y_err = abs(self.tgt_pos[1] - gl_tcp_pos[1])
        self.y_err.append(y_err)
        return self._y_limit - y_err

    def _constraint_z(self, jnt_values):
        """
        计算当前关节配置下,末端执行器的Z轴位置与目标位置之间的误差

        :param jnt_values: 当前关节角度
        :return: Z轴位置误差
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        z_err = abs(self.tgt_pos[2] - gl_tcp_pos[2])
        self.z_err.append(z_err)
        return self._z_limit - z_err

    def _constraint_collision(self, jnt_values):
        """
        检查当前关节配置下,机器人是否与障碍物发生碰撞

        :param jnt_values: 当前关节角度
        :return: 碰撞检测结果,-1表示碰撞,1表示无碰撞
        """
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        if self.rbt.is_collided(obstacle_list=self.obstacle_list):
            return -1
        else:
            return 1

    def add_constraint(self, fun, type="ineq"):
        """
        添加约束条件到优化问题中

        :param fun: 约束函数
        :param type: 约束类型,默认为不等式约束
        """
        self.cons.append({'type': type, 'fun': fun})

    def optimization_goal(self, jnt_values):
        """
        定义优化目标函数,最小化当前关节角度与种子关节角度之间的差异

        :param jnt_values: 当前关节角度
        :return: 当前关节角度与种子关节角度之间的差异
        """
        if self.toggle_debug:
            self.jnts.append(jnt_values)
            self.jnt_diff.append(np.linalg.norm(self.seed_jnt_values - jnt_values))
            # if random.choice(range(20)) == 0:
            #     self.rbth.show_armjnts(armjnts=self.jnts[-1], rgba=(.7, .7, .7, .2))
        return np.linalg.norm(jnt_values - self.seed_jnt_values)

    def solve(self, tgt_pos, tgt_rotmat, seed_jnt_values, method='SLSQP'):
        """
        通过优化求解逆运动学问题,找到使末端执行器达到目标位置和方向的关节角度

        :param tgt_pos: 目标位置
        :param tgt_rotmat: 目标旋转矩阵
        :param seed_jnt_values: 初始关节角度(种子值)
        :param method: 优化方法,默认为 'SLSQP'
        :return: 优化后的关节角度和目标函数值,如果失败则返回 (None, None)
        """
        self.seed_jnt_values = seed_jnt_values
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat

        # 添加各种约束条件
        self.add_constraint(self._constraint_xangle, type="ineq")
        self.add_constraint(self._constraint_zangle, type="ineq")
        self.add_constraint(self._constraint_x, type="ineq")
        self.add_constraint(self._constraint_y, type="ineq")
        self.add_constraint(self._constraint_z, type="ineq")
        self.add_constraint(self._constraint_collision, type="ineq")

        time_start = time.time()

        sol = minimize(self.optimization_goal,
                       seed_jnt_values,
                       method=method,
                       bounds=self.bnds,
                       constraints=self.cons)
        print("time cost", time.time() - time_start)  # 打印耗时

        if self.toggle_debug:
            print(sol)
            self._debug_plot()
        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

    def gen_linear_motion(self,
                          start_info,
                          goal_info,
                          granularity=0.03,
                          seed_jnt_values=None):
        """
        生成从起始位置到目标位置的线性运动路径

        :param start_info: 起始位置和旋转矩阵信息
        :param goal_info: 目标位置和旋转矩阵信息
        :param granularity: 插值粒度,默认为 0.03
        :param seed_jnt_values: 初始关节角度(种子值),如果为 None 则使用当前关节角度
        :return: 关节角度列表,如果无法求解则返回空列表
        author: weiwei
        date: 20210125
        """
        jnt_values_bk = self.rbt.get_jnt_values(self.jlc_name)  # 备份当前关节角度
        pos_list, rotmat_list = rm.interplate_pos_rotmat(start_info[0],
                                                         start_info[1],
                                                         goal_info[0],
                                                         goal_info[1],
                                                         granularity=granularity)
        jnt_values_list = []
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk

        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.solve(pos, rotmat, seed_jnt_values)[0]
            if jnt_values is None:
                print("无法求解！Not solvable!")  # 无法求解
                self.rbt.fk(self.jlc_name, jnt_values_bk)  # 恢复关节角度
                return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values

        self.rbt.fk(self.jlc_name, jnt_values_bk)  # 恢复关节角度
        return jnt_values_list

    def _debug_plot(self):
        """
        使用 Matplotlib 可视化优化过程中的误差和关节角度变化

        该方法会生成多个子图,分别展示位置误差、角度误差和关节角度变化
        """
        if "plt" not in dir():
            import visualization.matplot.helper as plt

        # 设置绘图窗口大小
        plt.plt.figure(1, figsize=(6.4 * 3, 4.8 * 2))

        # 绘制 x 方向误差
        plt.plt.subplot(231)
        plt.plt.plot(*plt.list_to_plt_xy(self.x_err))

        # 绘制 y 方向误差
        plt.plt.subplot(232)
        plt.plt.plot(*plt.list_to_plt_xy(self.y_err))

        # 绘制 z 方向误差
        plt.plt.subplot(233)
        plt.plt.plot(*plt.list_to_plt_xy(self.z_err))

        # 绘制 x 轴角度误差
        plt.plt.subplot(234)
        plt.plt.plot(*plt.list_to_plt_xy(self.xangle_err))

        # 绘制 z 轴角度误差
        plt.plt.subplot(235)
        plt.plt.plot(*plt.list_to_plt_xy(self.zangle_err))

        # 绘制关节角度变化
        plt.plt.subplot(236)
        plt.plt.plot(*plt.twodlist_to_plt_xys(self.jnts))
        # plt.plot_list(self.rot_err, title="rotation error")
        # plt.plot_list(self.jnt_diff, title="jnts displacement")

        # 显示绘图
        plt.plt.show()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim.robots.yumi.yumi as ym
    import modeling.geometric_model as gm
    import robot_sim.robots.gofa5.gofa5_dh76 as gofa5
    import robot_sim.end_effectors.gripper.dh76.dh76 as dh

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])

    tgt_pos = np.array([.5, -.3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    # yumi_instance = ym.Yumi(enable_cc=True)
    robot_s = gofa5.GOFA5(enable_cc=True)
    robot_s.gen_meshmodel().attach_to(base)
    component_name = 'arm'
    oik = FKOptBasedIK(robot_s, component_name=component_name, toggle_debug=False)

    # jnt_values, _ = oik.solve(tgt_pos, tgt_rotmat, np.zeros(7), method='SLSQP')
    # print(jnt_values)
    # robot_s.fk(hnd_name=hnd_name, jnt_values=jnt_values)
    # yumi_meshmodel = robot_s.gen_meshmodel()
    # yumi_meshmodel.attach_to(base)

    start_pos = np.array([.5, -.3, .3])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    start_info = [start_pos, start_rotmat]

    goal_pos = np.array([.6, .1, .5])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_info = [goal_pos, goal_rotmat]

    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)

    tic = time.time()
    jnt_values_list = oik.gen_linear_motion(start_info, goal_info)

    toc = time.time()
    print(toc - tic)

    for jnt_values in jnt_values_list:
        robot_s.fk(component_name, jnt_values)
        yumi_meshmodel = robot_s.gen_meshmodel()
        yumi_meshmodel.attach_to(base)
    base.run()
