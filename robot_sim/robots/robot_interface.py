import copy
import numpy as np
import robot_sim._kinematics.collision_checker as cc


class RobotInterface(object):
    """
    RobotInterface 类是一个机器人接口类,提供了与机器人各个组件(如操纵器、手部传感器等)进行交互的基本方法.
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        """
        初始化机器人接口对象,设置初始位置、旋转矩阵和名称
        :param pos: 机器人位置,默认为 3D 坐标原点 (0, 0, 0)
        :param rotmat: 机器人旋转矩阵,默认为单位矩阵
        :param name: 机器人接口的名称,默认为 'robot_interface'
        """
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cc = None
        # 组件字典,提供快速访问
        self.manipulator_dict = {}  # 操纵器字典
        self.ft_sensor_dict = {}  # 力传感器字典
        self.hnd_dict = {}  # 手部控制字典

    def change_name(self, name):
        """
        更改机器人的名称
        """
        self.name = name

    def get_hnd_on_manipulator(self, manipulator_name):
        """
        获取指定操纵器上的手部对象
        :param manipulator_name: 操纵器的名称
        :raises NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError

    def get_jnt_ranges(self, component_name):
        """
        获取指定组件的关节范围
        :param component_name: 组件名称
        :return: 关节范围
        """
        return self.manipulator_dict[component_name].get_jnt_ranges()

    def get_jnt_values(self, component_name):
        """
        获取指定组件的当前关节值
        :param component_name: 组件名称
        :return: 当前关节值
        """
        return self.manipulator_dict[component_name].get_jnt_values()

    def get_gl_tcp(self, manipulator_name):
        """
        获取指定操纵器的全局 TCP(Tool Center Point)位姿
        :param manipulator_name: 操纵器名称
        :return: 全局 TCP 坐标
        """
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        """
        检查给定的关节值是否在指定组件的关节范围内
        :param component_name: 组件名称
        :param jnt_values: 要检查的关节值
        :return: 如果关节值在范围内返回 True,否则返回 False
        """
        return self.manipulator_dict[component_name].is_jnt_values_in_ranges(jnt_values)

    def fix_to(self, pos, rotmat):
        """
        将机器人接口固定到指定位置和旋转矩阵
        此方法需要在子类中实现
        :param pos: 目标位置
        :param rotmat: 目标旋转矩阵
        :raises NotImplementedError: 该方法需要在子类中实现
        """
        return NotImplementedError

    def fk(self, component_name, jnt_values):
        """
        计算给定组件和关节值的正向运动学(Forward Kinematics)
        此方法需要在子类中实现
        :param component_name: 组件名称
        :param jnt_values: 关节值
        :raises NotImplementedError: 该方法需要在子类中实现
        """
        return NotImplementedError

    def jaw_to(self, hnd_name, jaw_width):
        """
        控制手部将其(Jaw)调到指定宽度
        :param hnd_name: 手部的名称
        :param jaw_width: 目标下巴宽度
        """
        self.hnd_dict[hnd_name].jaw_to(jaw_width=jaw_width)

    def get_jawwidth(self, hand_name):
        """
        获取指定手部的宽度
        :param hand_name: 手部的名称
        :return: 下巴宽度
        """
        return self.hnd_dict[hand_name].get_jawwidth()

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end",
           toggle_debug=False):
        """
        计算逆向运动学(IK),用于根据目标位置和姿态计算关节值
        :param component_name: 组件名称(默认 "arm")
        :param tgt_pos: 目标位置(默认为原点)
        :param tgt_rotmat: 目标旋转矩阵(默认为单位矩阵)
        :param seed_jnt_values: 初始的关节值(默认为 None)
        :param max_niter: 最大迭代次数(默认为 200)
        :param tcp_jnt_id: TCP 关节 ID(默认为 None)
        :param tcp_loc_pos: TCP 在局部坐标系中的位置(默认为 None)
        :param tcp_loc_rotmat: TCP 在局部坐标系中的旋转矩阵(默认为 None)
        :param local_minima: 局部极小值的处理策略,"end" 代表以末端为目标(默认为 "end")
        :param toggle_debug: 是否启用调试信息(默认为 False)
        :return: 计算出的关节值
        """
        return self.manipulator_dict[component_name].ik(tgt_pos,
                                                        tgt_rotmat,
                                                        seed_jnt_values=seed_jnt_values,
                                                        max_niter=max_niter,
                                                        tcp_jnt_id=tcp_jnt_id,
                                                        tcp_loc_pos=tcp_loc_pos,
                                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                                        local_minima=local_minima,
                                                        toggle_debug=toggle_debug)

    def manipulability(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       component_name='arm'):
        """
        计算给定组件(如机械臂)的操纵性指数(Manipulability Index)
        
        操纵性是衡量机器人是否能以最大的自由度进行运动的指标
        :param component_name: 组件名称(默认为 "arm")
        :return: 操纵性指数
        """
        return self.manipulator_dict[component_name].manipulability(tcp_jnt_id=tcp_jnt_id,
                                                                    tcp_loc_pos=tcp_loc_pos,
                                                                    tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id=None,
                             tcp_loc_pos=None,
                             tcp_loc_rotmat=None,
                             component_name='arm',
                             type="translational"):
        """
        计算给定组件(如机械臂)的操纵性轴向矩阵(Manipulability Axial Matrix)
        
        该矩阵表示机器人各个方向上的操纵性
        
        :param component_name: 组件名称(默认为 "arm")
        :param type: 操纵性的类型("translational" 或 "rotational")
        :return: 操纵性轴向矩阵
        """
        return self.manipulator_dict[component_name].manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                                                          tcp_loc_pos=tcp_loc_pos,
                                                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                                                          type=type)

    def jacobian(self,
                 component_name='arm',
                 tcp_jnt_id=None,
                 tcp_loc_pos=None,
                 tcp_loc_rotmat=None):
        """
        计算给定组件的雅可比矩阵(Jacobian Matrix)
        
        雅可比矩阵用于描述关节速度与末端执行器速度之间的关系
        
        :param component_name: 组件名称(默认为 "arm")
        :param tcp_jnt_id: TCP 关节 ID(默认为 None)
        :param tcp_loc_pos: TCP 在局部坐标系中的位置(默认为 None)
        :param tcp_loc_rotmat: TCP 在局部坐标系中的旋转矩阵(默认为 None)
        :return: 计算出的雅可比矩阵
        """
        return self.manipulator_dict[component_name].jacobian(tcp_jnt_id=tcp_jnt_id,
                                                              tcp_loc_pos=tcp_loc_pos,
                                                              tcp_loc_rotmat=tcp_loc_rotmat)

    def rand_conf(self, component_name):
        """
        生成一个随机的合法关节配置
        
        :param component_name: 组件名称
        :return: 随机关节配置
        """
        return self.manipulator_dict[component_name].rand_conf()

    def cvt_conf_to_tcp(self, manipulator_name, jnt_values):
        """
        将给定的关节值转换为对应的全局 TCP(Tool Center Point)位置和旋转矩阵
        
        :param manipulator_name: 操控器名称
        :param jnt_values: 关节值
        :return: 对应的全局 TCP 位置和旋转矩阵
        :author: weiwei
        :date: 20210417
        """
        jnt_values_bk = self.get_jnt_values(manipulator_name)  # 备份当前关节值
        self.robot_s.fk(manipulator_name, jnt_values)  # 根据给定的关节值计算正向运动学
        gl_tcp_pos, gl_tcp_rotmat = self.robot_s.get_gl_tcp(manipulator_name)  # 获取全局 TCP 位置和旋转矩阵
        self.robot_s.fk(manipulator_name, jnt_values_bk)  # 恢复原来的关节值
        return gl_tcp_pos, gl_tcp_rotmat

    def cvt_gl_to_loc_tcp(self, manipulator_name, gl_obj_pos, gl_obj_rotmat):
        """
        将全局坐标系中的物体位置和旋转矩阵转换为局部坐标系中的 TCP 位置和旋转矩阵
        
        :param manipulator_name: 操控器名称
        :param gl_obj_pos: 物体在全局坐标系中的位置
        :param gl_obj_rotmat: 物体在全局坐标系中的旋转矩阵
        :return: 转换后的局部 TCP 位置和旋转矩阵
        """
        return self.manipulator_dict[manipulator_name].cvt_gl_to_loc_tcp(gl_obj_pos, gl_obj_rotmat)

    def cvt_loc_tcp_to_gl(self, manipulator_name, rel_obj_pos, rel_obj_rotmat):
        """
        将局部坐标系中的物体位置和旋转矩阵转换为全局坐标系中的 TCP 位置和旋转矩阵
        
        :param manipulator_name: 操控器名称
        :param rel_obj_pos: 物体在局部坐标系中的位置
        :param rel_obj_rotmat: 物体在局部坐标系中的旋转矩阵
        :return: 转换后的全局 TCP 位置和旋转矩阵
        """
        return self.manipulator_dict[manipulator_name].cvt_loc_tcp_to_gl(rel_obj_pos, rel_obj_rotmat)

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contact_points=False):
        """
        检查机器人是否与障碍物或其他机器人发生碰撞
        
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param toggle_contact_points: 是否显示接触点(用于调试)
        :return: 碰撞信息,具体格式参见 CollisionChecker 的 is_collided 方法
        :author: weiwei
        :date: 20201223
        """
        if obstacle_list is None:
            obstacle_list = []
        if otherrobot_list is None:
            otherrobot_list = []
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             otherrobot_list=otherrobot_list,
                                             toggle_contact_points=toggle_contact_points)
        return collision_info

    def show_cdprimit(self):
        """
        显示碰撞检测的基本体(如包围盒、包围球等)
        """
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        """
        隐藏碰撞检测的基本体
        """
        self.cc.unshow_cdprimit()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        """
        生成机器人的简化模型(如棍状模型),用于可视化或碰撞检测
        
        该方法需要在子类中实现
        :param tcp_jnt_id: TCP 关节 ID
        :param tcp_loc_pos: TCP 在局部坐标系中的位置
        :param tcp_loc_rotmat: TCP 在局部坐标系中的旋转矩阵
        :param toggle_tcpcs: 是否显示 TCP 坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param toggle_connjnt: 是否显示连接关节
        :param name: 模型名称
        """
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        """
        生成机器人的详细模型(如网格模型),用于可视化或碰撞检测
        
        该方法需要在子类中实现
        
        :param tcp_jnt_id: TCP 关节 ID
        :param tcp_loc_pos: TCP 在局部坐标系中的位置
        :param tcp_loc_rotmat: TCP 在局部坐标系中的旋转矩阵
        :param toggle_tcpcs: 是否显示 TCP 坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param rgba: 模型的颜色和透明度
        :param name: 模型名称
        """
        raise NotImplementedError

    def enable_cc(self):
        """
        启用碰撞检测功能,初始化碰撞检测器
        """
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        禁用碰撞检测功能,清除碰撞对和节点路径
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        """
        深拷贝当前机器人对象,包括其所有属性和状态
        
        注意:由于碰撞检测器的深拷贝存在问题,需要手动更新
        :return: 机器人对象的深拷贝
        """
        self_copy = copy.deepcopy(self)
        # 深拷贝碰撞检测器时需要手动更新
        if self_copy.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
