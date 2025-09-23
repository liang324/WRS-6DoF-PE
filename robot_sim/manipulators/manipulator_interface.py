import copy
import numpy as np
import robot_sim._kinematics.collision_checker as cc
import modeling.model_collection as mc

from odio_urdf import *
from basis.robot_math import rotmat_to_euler

max_effort = 300
max_velocity = 10


def _link(N, robot_name, origin, mass, geom_origin):
    """
    大多数链接都是相同的,除了传入的信息.
    此函数只是为了更好地组织重要数据.
    """
    N = str(N)
    ret = Link(
        Inertial(
            Origin(xyz=origin[:3], rpy=origin[3:]),
            Mass(value=mass),
            Inertia(ixx=0, ixy=0, ixz=0, iyy=0, iyz=0, izz=0), ),  # 假设惯性矩阵为零
        Visual(Origin(geom_origin), ),  # 设置视觉原点
        Collision(Origin(geom_origin), ),  # 设置碰撞原点
        name=robot_name + "_link_" + N)  # 设置链接名称
    return ret


def _joint(N, robot_name, origin, limit, type="revolute"):
    """
    大多数关节都是相同的,除了传入的信息
    此函数只是为了更好地组织重要数据
    """
    N = int(N)
    ret = Joint(
        Parent(link=robot_name + "_link_" + str(N - 1)),  # 上一个链接
        Child(link=robot_name + "_link_" + str(N)),  # 当前链接
        Origin(xyz=origin[:3], rpy=origin[3:]),  # 设置原点的位置和姿态
        Axis(xyz="0 0 1"),  # 设置关节轴,这里假设是绕z轴旋转
        Limit(lower=limit[0], upper=limit[1], effort=max_effort, velocity=max_velocity),  # 设置关节的限位和约束
        type=type,  # 设置关节类型,默认为旋转关节
        name=robot_name + "_joint_" + str(N))  # 设置关节的名称
    return ret


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        """
        初始化机械臂接口
        :param pos: 机械臂的初始位置,默认为原点
        :param rotmat: 机械臂的初始旋转矩阵,默认为单位矩阵
        :param name: 机械臂的名称,默认为 'yumi_gripper'
        """
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # jlc: Joint and Link Configuration(关节和连杆配置)
        self.jlc = None
        # cc: Collision Checker(碰撞检测器)
        self.cc = None

    @property
    def jnts(self):
        """
        获取关节信息(Joint Information)
        """
        return self.jlc.jnts

    @property
    def lnks(self):
        """
        获取连杆信息(Link Information)
        """
        return self.jlc.lnks

    @property
    def tgtjnts(self):
        """
        获取目标关节序列(Target Joints)
        """
        return self.jlc.tgtjnts

    @property
    def ndof(self):
        """
        获取自由度数量(Number of Degrees of Freedom)
        """
        return self.jlc.ndof

    @property
    def homeconf(self):
        """
        获取机械臂的初始姿态(Home Configuration)
        """
        return self.jlc.homeconf

    @property
    def tcp_jnt_id(self):
        """
        获取末端执行器的关节 ID(TCP: Tool Center Point)
        """
        return self.jlc.tcp_jnt_id

    @property
    def tcp_loc_pos(self):
        """
        获取末端执行器在局部坐标系中的位置(Local Position of TCP)
        """
        return self.jlc.tcp_loc_pos

    @property
    def tcp_loc_rotmat(self):
        """
        获取末端执行器在局部坐标系中的旋转矩阵(Local Rotation of TCP)
        """
        return self.jlc.tcp_loc_rotmat

    @tcp_jnt_id.setter
    def tcp_jnt_id(self, tcp_jnt_id):
        """
        设置末端执行器的关节 ID(Set TCP Joint ID)
        """
        self.jlc.tcp_jnt_id = tcp_jnt_id

    @tcp_loc_pos.setter
    def tcp_loc_pos(self, tcp_loc_pos):
        """
        设置末端执行器在局部坐标系中的位置(Set TCP Local Position)
        """
        self.jlc.tcp_loc_pos = tcp_loc_pos

    @tcp_loc_rotmat.setter
    def tcp_loc_rotmat(self, tcp_loc_rotmat):
        """
        设置 TCP 的局部旋转矩阵
        :param tcp_loc_rotmat: 3x3 旋转矩阵
        """
        self.jlc.tcp_loc_rotmat = tcp_loc_rotmat

    def set_homeconf(self, jnt_values):
        """
        设置机械臂的 Home Configuration(初始姿态)
        :param jnt_values: 关节角度列表
        """
        self.jlc.set_homeconf(jnt_values=jnt_values)

    def set_tcp(self, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        一次性设置 TCP(末端执行器)的关节ID、局部位置与朝向
        :param tcp_jnt_id: TCP 所在的关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        """
        if tcp_jnt_id is not None:
            self.jlc.tcp_jnt_id = tcp_jnt_id
        if tcp_loc_pos is not None:
            self.jlc.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.jlc.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self,
                   tcp_jnt_id=None,
                   tcp_loc_pos=None,
                   tcp_loc_rotmat=None):
        """
        获取当前 TCP 在全局坐标系中的位置和旋转
        :return: (位置, 旋转矩阵)
        """
        return self.jlc.get_gl_tcp(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat)

    def get_jnt_ranges(self):
        """
        获取各个关节的运动范围(上下限)
        :return: 关节范围列表
        """
        return self.jlc._get_jnt_ranges()

    def goto_homeconf(self):
        """
        将机械臂移动到初始 Home Configuration 姿态
        """
        self.jlc.fk(jnt_values=self.jlc.homeconf)

    def goto_zeroconf(self):
        """
        将机械臂移动到 Zero Configuration(关节值全为0)
        """
        self.jlc.fk(jnt_values=self.jlc.zeroconf)

    def fix_to(self, pos, rotmat, jnt_values=None):
        """
        将机械臂固定在某个位置和姿态下(通常用于可视化或仿真)
        :param pos: 全局位置
        :param rotmat: 全局旋转矩阵
        :param jnt_values: 对应的关节角度
        :return: 更新后的状态
        """
        return self.jlc.fix_to(pos=pos, rotmat=rotmat, jnt_values=jnt_values)

    def is_jnt_values_in_ranges(self, jnt_values):
        """
        判断给定关节角度是否在合法范围内
        :param jnt_values: 待检测的关节值
        :return: True/False
        """
        return self.jlc.is_jnt_values_in_ranges(jnt_values)

    def fk(self, jnt_values):
        """
        正向运动学(Forward Kinematics)
        :param jnt_values: 指定的关节值
        :return: 更新姿态后的位置和朝向
        """
        return self.jlc.fk(jnt_values=jnt_values)

    def get_jnt_values(self):
        """
        获取当前的关节角度列表
        :return: 关节值数组
        """
        return self.jlc.get_jnt_values()

    def rand_conf(self):
        """
        获取一个随机的有效关节姿态(用于采样或规划)
        :return: 随机姿态关节值数组
        """
        return self.jlc.rand_conf()

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           max_niter=100,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        """
        逆向运动学(Inverse Kinematics):计算使末端到达目标位置的关节角度
        :param tgt_pos: 目标位置
        :param tgt_rotmat: 目标旋转矩阵
        :param seed_jnt_values: 初始猜测的关节值(用于迭代)
        :param max_niter: 最大迭代次数
        :param tcp_jnt_id: TCP 所在关节ID(可选)
        :param tcp_loc_pos: TCP 的局部位置(可选)
        :param tcp_loc_rotmat: TCP 的局部旋转(可选)
        :param local_minima: 是否接受局部最优解("accept" 或 "reject")
        :param toggle_debug: 是否开启调试模式
        :return: 解算后的关节值(或 None)
        """
        return self.jlc.ik(tgt_pos=tgt_pos,
                           tgt_rotmat=tgt_rotmat,
                           seed_jnt_values=seed_jnt_values,
                           max_niter=max_niter,
                           tcp_jnt_id=tcp_jnt_id,
                           tcp_loc_pos=tcp_loc_pos,
                           tcp_loc_rotmat=tcp_loc_rotmat,
                           local_minima=local_minima,
                           toggle_debug=toggle_debug)

    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        """
        计算操控性(Manipulability),评估末端执行器的运动能力
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :return: 操控性值
        """
        return self.jlc.manipulability(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat, type="translational"):
        """
        计算操控性的雅可比矩阵(或称运动能力矩阵)
        :param type: 计算类型,默认为 "translational",即平移操作
        :return: 雅可比矩阵
        """
        return self.jlc.manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                             tcp_loc_pos=tcp_loc_pos,
                                             tcp_loc_rotmat=tcp_loc_rotmat,
                                             type=type)

    def jacobian(self,
                 tcp_jnt_id,
                 tcp_loc_pos,
                 tcp_loc_rotmat):
        """
        计算雅可比矩阵,表示关节变动对末端执行器的影响
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :return: 雅可比矩阵
        """
        return self.jlc.jacobian(tcp_jnt_id=tcp_jnt_id,
                                 tcp_loc_pos=tcp_loc_pos,
                                 tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_loc_tcp_to_gl(self,
                          loc_pos=np.zeros(3),
                          loc_rotmat=np.eye(3),
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        将TCP的局部位置和旋转矩阵转换为全局坐标系下的表示
        :param loc_pos: TCP 局部位置
        :param loc_rotmat: TCP 局部旋转矩阵
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :return: 全局位置和旋转矩阵
        """
        return self.jlc.cvt_loc_tcp_to_gl(loc_pos=loc_pos,
                                          loc_rotmat=loc_rotmat,
                                          tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_gl_to_loc_tcp(self,
                          gl_pos,
                          gl_rotmat,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        将全局坐标系下的TCP位置和旋转矩阵转换为局部坐标系下的表示
        :param gl_pos: 全局位置
        :param gl_rotmat: 全局旋转矩阵
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :return: 局部位置和旋转矩阵
        """
        return self.jlc.cvt_gl_to_loc_tcp(gl_pos=gl_pos,
                                          gl_rotmat=gl_rotmat,
                                          tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        检查是否发生碰撞,接口方法,需要在子类中实现
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :return: 碰撞状态(True 或 False)
        author: weiwei
        date: 20201223
        """
        return self.cc.is_collided(obstacle_list=obstacle_list,
                                   otherrobot_list=otherrobot_list)

    def show_cdprimit(self):
        """
        显示碰撞原始体(Collision Primitive).用于调试或可视化碰撞体.
        """
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        """
        隐藏碰撞原始体(Collision Primitive).用于调试或可视化碰撞体.
        """
        self.cc.unshow_cdprimit()

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      rgba=None,
                      name='manipulator_mesh'):
        """
        生成机械臂的网格模型.可以通过设置 TCP 位置、旋转矩阵来定义末端执行器的位置
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :param toggle_tcpcs: 是否显示 TCP 坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param rgba: 颜色,RGBA 格式
        :param name: 模型的名称
        :return: 生成的网格模型
        """
        return self.jlc._mt.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=toggle_tcpcs,
                                          toggle_jntscs=toggle_jntscs,
                                          name=name, rgba=rgba)

    def gen_stickmodel(self,
                       rgba=np.array([.5, 0, 0, 1]),
                       thickness=.01,
                       joint_ratio=1.62,
                       link_ratio=.62,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=True,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='jlcstick'):
        """
        生成机械臂的棍棒模型.该模型通常用于可视化显示机械臂的关节和连接
        :param rgba: 颜色,RGBA 格式
        :param thickness: 棍棒的厚度
        :param joint_ratio: 关节比率
        :param link_ratio: 连杆比率
        :param tcp_jnt_id: TCP 所在关节ID
        :param tcp_loc_pos: TCP 相对于关节坐标系的局部位置
        :param tcp_loc_rotmat: TCP 相对于关节坐标系的局部旋转矩阵
        :param toggle_tcpcs: 是否显示 TCP 坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param toggle_connjnt: 是否显示关节连接
        :param name: 模型的名称
        :return: 生成的棍棒模型
        """
        return self.jlc._mt.gen_stickmodel(rgba=rgba,
                                           thickness=thickness,
                                           joint_ratio=joint_ratio,
                                           link_ratio=link_ratio,
                                           tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=toggle_tcpcs,
                                           toggle_jntscs=toggle_jntscs,
                                           toggle_connjnt=toggle_connjnt,
                                           name=name)

    def gen_endsphere(self):
        """
        生成末端球体模型,用于表示末端执行器的最终位置
        :return: 生成的末端球体模型
        """
        return self.jlc._mt.gen_endsphere()

    def enable_cc(self):
        """
        启用碰撞检测(Collision Checking)功能.
        创建一个碰撞检查器实例,用于后续的碰撞检测.
        """
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        禁用碰撞检测功能,清除所有碰撞检查相关的数据
        清除所有碰撞元素并将 `self.cc` 设置为 `None`,结束碰撞检测功能
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1  # 清空碰撞元素的关联ID
        self.cc = None
        # self.cc.all_cdelements = []
        # for child in self.cc.np.getChildren():
        #     child.removeNode()
        # self.cc.nbitmask = 0

    def copy(self):
        """
        复制当前对象,并在复制后的对象中手动修正碰撞检测的关联.
        因为深拷贝的碰撞检测器需要手动重新关联其碰撞元素.
        :return: 复制的对象
        """
        self_copy = copy.deepcopy(self)
        # 复制碰撞体时需要手动更新碰撞器
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():  # 如果没有子节点则清空 NodePathCollection
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy

    def gen_urdf(self):
        """
        生成机器人模型的 URDF 文件,描述机器人的结构、关节、连杆、质量等信息.
        :return: 生成的 URDF 文件内容
        """
        parent = 'world'  # 世界坐标系作为根节点
        robot_name = self.name  # 机器人名称
        WORLD_LNK = Link(name=parent, )  # 创建世界链接
        group = []  # 用于存储所有关节和连杆的列表
        # 遍历机器人的每个关节,生成 URDF 的关节和连杆描述
        for i in range(self.jlc.ndof + 1):  # `self.jlc.ndof` 表示机器人自由度
            if i == 0:  # 对于根节点(世界坐标系)
                # 创建一个固定关节
                jnt = Joint(parent + "_" + robot_name + "_joint",  # 关节名称
                            Origin(xyz="0 0 0", rpy="0 0 0"),  # 坐标原点与旋转角度
                            Parent(link=parent),  # 父链接(世界坐标系)
                            Child(link=robot_name + "_link_0"),  # 子链接(机器人第一个连杆)
                            type="fixed")  # 关节类型为固定关节
            else:
                # 对于其他关节,调用 `_joint` 函数生成
                jnt = _joint(
                    i,  # 关节的编号
                    robot_name,  # 机器人名称
                    origin=[*self.jnts[i]['loc_pos'], *rotmat_to_euler(self.jnts[i]['loc_rotmat'])],  # 关节位置与旋转(转换为欧拉角)
                    limit=self.jnts[i]['motion_rng'],  # 关节的运动范围
                    type=self.jnts[i]['type'])  # 关节类型
            group.append(jnt)  # 将关节添加到组中
            # 为每个关节生成对应的连杆(Link)
            lnk = _link(
                i,  # 连杆的编号
                robot_name,  # 机器人名称
                origin=[*self.lnks[i]['loc_pos'], *rotmat_to_euler(self.lnks[i]['loc_rotmat'])],  # 连杆的位置与旋转
                # x, y, z, r, p, y
                mass=self.lnks[i]['mass'],  # 连杆的质量
                geom_origin=self.lnks[i]['com'].tolist())  # 连杆的质心
            group.append(lnk)  # 将连杆添加到组中

        # 创建 URDF 机器人对象
        URDF = Robot(WORLD_LNK, *group, name=self.name)  # 生成机器人URDF对象
        return str(URDF)  # 返回 URDF 文件内容(字符串)
