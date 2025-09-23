# 这段代码定义了一个 GripperInterface 类,它是一个机械夹爪的接口,用于控制夹爪的运动学、碰撞检测以及与物体的交互.
# 该类包含了许多用于操作夹爪的方法,如设置夹爪的开口宽度、计算夹爪的运动、检测碰撞等.

import copy
import os
import numpy as np
import modeling.model_collection as mc
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim._kinematics.collision_checker as cc
from basis.trimesh import base


class GripperInterface(object):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='aabb', name='gripper'):
        """
        :param pos: 夹爪的初始位置,默认值是原点 [0, 0, 0]
        :param rotmat: 夹爪的初始旋转矩阵,默认为单位矩阵,表示没有旋转
        :param cdmesh_type: 设置碰撞检测模型的类型(例如:aabb、convexhull 或 triangles)
        :param name: 夹爪的名称,默认为 'gripper'
        """
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cdmesh_type = cdmesh_type  # 碰撞检测模型的类型(aabb、convexhull 或 triangles)

        # joints默认不耦合
        # 通过 JLChain 创建夹爪的耦合链(机械臂的运动链),它初始化了夹爪的初始位置、旋转矩阵以及一些默认配置(如homeconf,通常是关节的初始配置)
        self.coupling = jl.JLChain(pos=self.pos, rotmat=self.rotmat, homeconf=np.zeros(0), name='coupling')
        # 为夹爪的关节和链接指定了默认的物理属性(如位置和名称),并重新初始化耦合链
        self.coupling.jnts[1]['loc_pos'] = np.array([0, 0, .0])
        self.coupling.lnks[0]['name'] = 'coupling_lnk0'
        # 为耦合链中的链接(link)分配显式网格模型
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "xxx.stl")
        self.coupling.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.coupling.reinitialize()
        # 钳爪中心
        self.jaw_center_pos = np.zeros(3)
        self.jaw_center_rotmat = np.eye(3)
        # 钳爪宽度
        self.jawwidth_rng = [0.0, 5.0]
        # 碰撞检测
        self.cc = None
        # 用于精确碰撞检测的碰撞网格集合
        self.cdmesh_collection = mc.ModelCollection()

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        是否发生碰撞”的接口,必须在子类中实现.
        :param obstacle_list: 障碍物列表.
        :param otherrobot_list: 其他机器人列表.
        :return: 如果发生碰撞,返回 True；否则,返回 False.
        author: weiwei
        date: 20201223
        """
        return_val = self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        return return_val

    def is_mesh_collided(self, objcm_list=[], toggle_debug=False):
        """
        检测碰撞网格集合与给定对象列表之间是否存在碰撞.

        :param objcm_list: 用于碰撞检测的碰撞网格对象列表. 默认为 [].
        :param toggle_debug: 是否启用调试模式. 如果启用, 将打印碰撞信息并显示碰撞网格和碰撞点. 默认为 False.
        :return: 如果检测到碰撞, 返回 True; 否则, 返回 False.
        """
        for i, cdelement in enumerate(self.all_cdelements):
            # 从碰撞元素中提取位置和旋转矩阵
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']

            # 设置碰撞网格的位置和旋转
            self.cdmesh_collection.cm_list[i].set_pos(pos)  # 将碰撞网格的位置设置为当前碰撞元素的位置
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)  # 将碰撞网格的旋转设置为当前碰撞元素的旋转矩阵

            # 检测当前碰撞网格与给定对象列表之间的碰撞
            iscollided, collided_points = self.cdmesh_collection.cm_list[i].is_mcdwith(objcm_list, True)
            if iscollided:
                if toggle_debug:
                    print(self.cdmesh_collection.cm_list[i].get_homomat())
                    # 显示当前碰撞网格
                    self.cdmesh_collection.cm_list[i].show_cdmesh()
                    # 显示所有参与碰撞检测的对象网格
                    for objcm in objcm_list:
                        objcm.show_cdmesh()
                    # 为每个碰撞点生成一个小球以可视化碰撞位置
                    for point in collided_points:
                        import modeling.geometric_model as gm
                        gm.gen_sphere(point, radius=.001).attach_to(base)
                    print("collided")
                return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def fk(self, motion_val):
        raise NotImplementedError

    def jaw_to(self, jaw_width):
        raise NotImplementedError

    def get_jawwidth(self):
        raise NotImplementedError

    def grip_at_with_jczy(self, gl_jaw_center_pos, gl_jaw_center_z, gl_jaw_center_y, jaw_width):
        """
        计算夹爪的旋转矩阵并执行夹持操作

        :param gl_jaw_center_pos: 夹爪中心的位置 (3D向量)
        :param gl_jaw_center_z: 夹爪中心的接近方向 (3D向量)
        :param gl_jaw_center_y: 夹爪中心的打开方向 (3D向量)
        :param jaw_width: 夹爪的宽度
        :return: 调用 grip_at_with_jcpose 函数的结果

        """
        # 初始化旋转矩阵为单位矩阵
        gl_jaw_center_rotmat = np.eye(3)
        # 设置旋转矩阵的第三列为夹爪中心的接近方向的单位向量
        gl_jaw_center_rotmat[:, 2] = rm.unit_vector(gl_jaw_center_z)
        # 设置旋转矩阵的第二列为夹爪中心的打开方向的单位向量
        gl_jaw_center_rotmat[:, 1] = rm.unit_vector(gl_jaw_center_y)
        # 通过叉积计算旋转矩阵的第一列,确保正交性
        gl_jaw_center_rotmat[:, 0] = np.cross(gl_jaw_center_rotmat[:3, 1], gl_jaw_center_rotmat[:3, 2])
        # 调用 grip_at_with_jcpose 函数,传入计算得到的旋转矩阵和其他参数
        return self.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)

    def grip_at_with_jcpose(self, gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width):
        """
        根据给定的夹爪中心位置、旋转矩阵和宽度,计算并设置末端执行器的位置和方向
        :param gl_jaw_center_pos: 夹爪中心在全局坐标系中的位置(通常是一个三维坐标)
        :param gl_jaw_center_rotmat: 夹爪中心的旋转矩阵,表示夹爪的朝向
        :param jaw_width: 夹爪的开口宽度,表示夹爪张开的程度
        :return:
        """
        # 设置夹爪宽度
        self.jaw_to(jaw_width)
        # 计算末端执行器的旋转矩阵
        eef_root_rotmat = gl_jaw_center_rotmat.dot(self.jaw_center_rotmat.T)
        # 计算末端执行器的位置
        eef_root_pos = gl_jaw_center_pos - eef_root_rotmat.dot(self.jaw_center_pos)
        # 固定末端执行器的位置和方向
        self.fix_to(eef_root_pos, eef_root_rotmat)
        # 包含夹爪宽度、全局夹爪中心位置、全局夹爪中心旋转矩阵、末端执行器位置和末端执行器旋转矩阵
        return [jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]

    def show_cdprimit(self):
        """
        显示碰撞检测的基本形状(原语)
        """
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        """
        隐藏碰撞检测的基本形状(原语)
        """
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        """
        显示碰撞检测的网格模型
        """
        # 遍历所有的碰撞元素
        for i, cdelement in enumerate(self.cc.all_cdelements):
            pos = cdelement['gl_pos']  # 获取全局位置
            rotmat = cdelement['gl_rotmat']  # 获取全局旋转矩阵
            # 设置网格模型的位置和旋转矩阵
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        # 显示网格模型
        self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        """
        隐藏碰撞检测的网格模型
        """
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='gripper_stickmodel'):
        """
        生成简化的杆状模型
        """
        raise NotImplementedError

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='gripper_meshmodel'):
        """
        生成详细的网格模型
        """
        raise NotImplementedError

    def enable_cc(self):
        """
        启用碰撞检测
        """
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        禁用碰撞检测,清除配对和节点路径

        :return:
        """
        # 将所有碰撞元素的子节点ID设置为-1
        for cdelement in self.cc.all_cdelements:
            # 将碰撞检测对象设置为None
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        """
        创建对象的深拷贝
        """
        self_copy = copy.deepcopy(self)
        # 深拷贝碰撞器有问题,需要手动更新
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
