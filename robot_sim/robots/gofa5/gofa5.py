import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.gofa5_arm.gofa5_arm as rbt
import robot_sim.end_effectors.gripper.ag145.ag145 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import copy


class GOFA5(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="gofa5", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_stand = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand')

        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "table.stl"),
            cdprimit_type="box", expand_radius=.0001,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand.reinitialize()
        self.base_stand2 = jl.JLChain(pos=pos,
                                      rotmat=rotmat,
                                      homeconf=np.zeros(0),
                                      name='base_stand2')

        self.base_stand2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "connect.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand2.reinitialize()

        self.camera1 = jl.JLChain(pos=pos,
                                  rotmat=rotmat,
                                  homeconf=np.zeros(0),
                                  name='camera1')

        self.camera1.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "camera1.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.camera1.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.camera1.reinitialize()

        self.camera2 = jl.JLChain(pos=pos,
                                  rotmat=rotmat,
                                  homeconf=np.zeros(0),
                                  name='camera2')

        self.camera2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "camera2.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.camera2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.camera2.reinitialize()

        self.camera3 = jl.JLChain(pos=pos,
                                  rotmat=rotmat,
                                  homeconf=np.zeros(0),
                                  name='camera3')

        self.camera3.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "camera3.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.camera3.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.camera3.reinitialize()

        # arm
        arm_homeconf = np.zeros(6)
        # self.arm = rbt.Gofa5Arm(pos=pos + [0.13747, -0.03748, 0.015],
        #                         rotmat=self.base_stand.jnts[-1]['gl_rotmatq'],
        #                         homeconf=arm_homeconf,
        #                         name='arm', enable_cc=False)
        self.arm = rbt.Gofa5Arm(homeconf=arm_homeconf,
                                name='arm', enable_cc=False)
        # gripper
        self.hnd = hnd.Ag145(pos=self.arm.jnts[-1]['gl_posq'],
                             rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                             name='hnd', enable_cc=False)

        self.brand = cm.CollisionModel(os.path.join(this_dir, "meshes", "logo_01.stl"))
        self.brand.set_rgba([1, 0, 0, 1])
        self.brand.set_pos(self.arm.jnts[2]['gl_posq'])
        self.brand.set_rotmat(self.arm.jnts[2]['gl_rotmatq'])

        self.brand_arm = cm.CollisionModel(os.path.join(this_dir, "meshes", "logo_02.stl"))
        self.brand_arm.set_rgba([1, 0, 0, 1])
        self.brand_arm.set_pos(self.arm.jnts[4]['gl_posq'])
        self.brand_arm.set_rotmat(self.arm.jnts[4]['gl_rotmatq'])

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.1, 0.0, 0.14 - 0.82),
                                              x=.35 + radius, y=.3 + radius, z=.14 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3),
                                              x=.112 + radius, y=.112 + radius, z=.3 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.base_stand2, [0])
        self.cc.add_cdlnks(self.camera1, [0])
        self.cc.add_cdlnks(self.camera2, [0])
        self.cc.add_cdlnks(self.camera3, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2, 3])
        self.cc.add_cdlnks(self.hnd.lft_inner, [1])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2, 3])
        self.cc.add_cdlnks(self.hnd.rgt_inner, [1])
        self.cc.add_cdlnks(self.hnd.lft_inner_connectingrod, [1, 2, 3])
        self.cc.add_cdlnks(self.hnd.rgt_inner_connectingrod, [1, 2, 3])
        activelist = [self.base_stand.lnks[0],
                      self.base_stand2.lnks[0],
                      self.camera1.lnks[0],
                      self.camera2.lnks[0],
                      self.camera3.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.lft_outer.lnks[3],
                      self.hnd.lft_inner.lnks[1],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[3],
                      self.hnd.rgt_inner.lnks[1],
                      self.hnd.lft_inner_connectingrod.lnks[1],
                      self.hnd.lft_inner_connectingrod.lnks[2],
                      self.hnd.lft_inner_connectingrod.lnks[3],
                      self.hnd.rgt_inner_connectingrod.lnks[1],
                      self.hnd.rgt_inner_connectingrod.lnks[2],
                      self.hnd.rgt_inner_connectingrod.lnks[3]]

        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.base_stand.lnks[0],
                    self.base_stand2.lnks[0],
                    self.arm.lnks[1],
                    self.camera1.lnks[0],
                    self.camera2.lnks[0],
                    self.camera3.lnks[0]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_inner.lnks[1],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_inner.lnks[1],
                    self.hnd.lft_inner_connectingrod.lnks[1],
                    self.hnd.lft_inner_connectingrod.lnks[2],
                    self.hnd.lft_inner_connectingrod.lnks[3],
                    self.hnd.rgt_inner_connectingrod.lnks[1],
                    self.hnd.rgt_inner_connectingrod.lnks[2],
                    self.hnd.rgt_inner_connectingrod.lnks[3]]

        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'], rotmat=self.base_stand.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])

        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    # def ik(self,
    #        component_name: str = "arm",
    #        tgt_pos=np.zeros(3),
    #        tgt_rotmat=np.eye(3),
    #        seed_jnt_values=None,
    #        max_niter=200,
    #        num_seeds=2,  # 设置seed数量
    #        tcp_jnt_id=None,
    #        tcp_loc_pos=None,
    #        tcp_loc_rotmat=None,
    #        local_minima: str = "end",
    #        toggle_debug=False):
    #     if component_name == "arm":
    #         directory = os.path.abspath(os.path.dirname(__file__))
    #
    #         grandparent_directory = os.path.abspath(os.path.join(directory, "..", ".."))
    #         # 双引号后面加入机械臂的文件夹位置
    #         directory = os.path.abspath(os.path.join(grandparent_directory, "manipulators/gofa5"))
    #         # 双引号后面加入机械臂的文件名
    #         yumi_urdf = os.path.join(directory, "gofa5_arm.urdf")
    #         ik_solver = TracIK("world", "gofa5_arm_link_6", yumi_urdf)
    #         # ik_solver = TracIK("world", "tbm_link_7", yumi_urdf,solver_type="Manip2")
    #         tgt_pos = tgt_pos - np.dot(tgt_rotmat,
    #                                    self.hnd_dict[component_name].jaw_center_pos)
    #         seed = np.array([0, 0, 0, 0, 0, 0])
    #         solution = ik_solver.ik(tgt_pos, tgt_rotmat, seed)
    #         print('solution', solution)
    #         return solution

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        计算给定组件的正向运动学(FK)

        :param jnt_values: 关节值,可以是 7 或 3+7,3=agv,7=arm,1=grpr；单位为米-弧度
        :param component_name: 组件名称,可以是 'arm'、'agv' 或 'all'
        :return: 返回正向运动学计算的状态
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            # 更新对象信息的全局位置和旋转矩阵
            for obj_info in self.oih_infos:
                # 将局部位置和旋转矩阵转换为全局坐标系
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos  # 更新全局位置
                obj_info['gl_rotmat'] = gl_rotmat  # 更新全局旋转矩阵

        def update_component(component_name, jnt_values):
            # 计算给定关节值的正向运动学
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)

            # 固定手柄到末端执行器的位置和旋转
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            # 更新品牌的位置和旋转矩阵
            self.brand.set_pos(self.manipulator_dict[component_name].jnts[2]['gl_posq'])
            self.brand.set_rotmat(self.manipulator_dict[component_name].jnts[2]['gl_rotmatq'])
            self.brand_arm.set_pos(self.manipulator_dict[component_name].jnts[4]['gl_posq'])
            self.brand_arm.set_rotmat(self.manipulator_dict[component_name].jnts[4]['gl_rotmatq'])
            # 更新对象信息
            update_oih(component_name=component_name)
            return status  # 返回正向运动学计算的状态

        # 检查组件名称是否在操纵器字典中
        if component_name in self.manipulator_dict:
            # 检查关节值是否为 1x6 的 numpy 数组
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)  # 更新组件
        else:
            raise ValueError("The given component name is not supported!")  # 抛出错误

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    # def get_jnt_init(self, component_name):
    #     if component_name in self.manipulator_dict:
    #         return self.arm.init_jnts
    #     else:
    #         raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hand_name, jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.base_stand.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            # print(jawwidth)
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.base_stand2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.camera1.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.camera2.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.camera3.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      is_machine=None,
                      is_robot=True,
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=False,
                                          toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.base_stand2.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=False,
                                           toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.camera1.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
            self.camera2.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
            self.camera3.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
            brand = copy.deepcopy(self.brand)
            brand.attach_to(meshmodel)
            brand_arm = copy.deepcopy(self.brand_arm)
            brand_arm.attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           num_seeds=2,  # 设置seed数量
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end",
           toggle_debug=False):
        if component_name == "arm" or component_name == "hnd":
            # ik_solver = TracIK("world", "tbm_link_7", yumi_urdf,solver_type="Manip2")
            tgt_pos = tgt_pos - np.dot(tgt_rotmat,
                                       self.hnd_dict[component_name].jaw_center_pos)
            # seed = np.array([0, 0, 0, 0, 0, 0])
            ik_solution = self.arm.ik(tgt_pos=tgt_pos,
                                      tgt_rotmat=tgt_rotmat,
                                      seed_jnt_values=seed_jnt_values,
                                      max_niter=max_niter,
                                      tcp_jntid=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      local_minima=local_minima,
                                      toggle_debug=toggle_debug)
            # print('solution', solution)
            return ik_solution


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])

    gm.gen_frame(length=.6).attach_to(base)
    robot_s = GOFA5(enable_cc=True)
    # robot_s.hnd.jaw_to(.06)
    # robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    # robot_s.hnd.jaw_to(.14)
    # robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    tgt_pos = np.array([ 0.81905066, -0.19977778,  0.07733738])
    tgt_rotmat = np.array([[7.07106781e-01, 8.67361672e-17, 7.07106781e-01],
                           [6.13317320e-17, -1.00000000e+00, 6.13317320e-17],
                           [7.07106781e-01, 0.00000000e+00, -7.07106781e-01]])
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, length=.3).attach_to(base)
    # robot_s.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    # base.run()
    component_name = 'arm'
    st = time.time()
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat, )
    print("IK's time consumption is", time.time() - st)
    print(jnt_values)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    robot_s_meshmodel.attach_to(base)
    # # robot_s.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    result, ps = robot_s.is_collided(toggle_contact_points=True)
    for p in ps:
        gm.gen_sphere(p, radius=0.06).attach_to(base)

    # toc = time.time()
    print(result)
    base.run()
