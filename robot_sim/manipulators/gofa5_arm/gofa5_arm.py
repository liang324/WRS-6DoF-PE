import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import modeling.collision_model as cm
import modeling.model_collection as mc

try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Track IK starts")
except Exception as e:
    print("Track IK not available")
    print(e)
    is_trac_ik = False


class Gofa5Arm(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='gofa5_arm', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        jnt_safemargin = math.pi / 18.0

        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.1855])
        self.jlc.jnts[1]['motion_rng'] = [-math.pi / 2 + jnt_safemargin, math.pi / 2 - jnt_safemargin]

        self.jlc.jnts[2]['loc_pos'] = np.array([0, -.085, 0.0765])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=math.pi)
        self.jlc.jnts[2]['motion_rng'] = [-math.pi + jnt_safemargin, math.pi - jnt_safemargin]
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0.444, 0])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(ai=0, aj=0, ak=0)
        self.jlc.jnts[3]['motion_rng'] = [-225 * (math.pi / 180) + jnt_safemargin,
                                          85 * (math.pi / 180) - jnt_safemargin]
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[4]['loc_pos'] = np.array([-0.096, 0.11, 0.085])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi * 0 / 2, aj=math.pi / 2 - math.pi,
                                                              ak=math.pi * 0 / 2)
        self.jlc.jnts[4]['motion_rng'] = [-math.pi + jnt_safemargin, math.pi - jnt_safemargin]
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[5]['loc_pos'] = np.array([0.0755, 0, 0.373])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(ai=0, aj=math.pi / 2 - math.pi, ak=math.pi)
        self.jlc.jnts[5]['motion_rng'] = [-math.pi + jnt_safemargin, math.pi - jnt_safemargin]
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[6]['loc_pos'] = np.array([0.101, -0.08, -0.0745])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(ai=-math.pi * 1 / 2, aj=math.pi * 0 / 2,
                                                              ak=-math.pi * 1 / 2)
        self.jlc.jnts[6]['motion_rng'] = [-1.5 * math.pi + jnt_safemargin, 1.5 * math.pi - jnt_safemargin]
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])

        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 2.0
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK00.STL")
        self.jlc.lnks[0]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[1]['name'] = "shoulder"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.jlc.lnks[1]['mass'] = 1.95
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK01.STL")
        self.jlc.lnks[1]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[2]['name'] = "upperarm"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.jlc.lnks[2]['mass'] = 3.42
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK02.STL")
        self.jlc.lnks[2]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[3]['name'] = "forearm"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.jlc.lnks[3]['mass'] = 1.437
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK03.STL")
        self.jlc.lnks[3]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[4]['name'] = "wrist1"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[4]['mass'] = 0.871
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK04.STL")
        self.jlc.lnks[4]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[5]['name'] = "wrist2"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[5]['mass'] = 0.8
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK05.STL")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[6]['name'] = "wrist3"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, 0])
        self.jlc.lnks[6]['mass'] = 0.8
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK06.STL")
        self.jlc.lnks[6]['rgba'] = [.7, .7, .7, 1]
        self.jlc.reinitialize()

        # 添加 ABB logo
        self.logo_02 = jl.JLChain(pos=self.jlc.jnts[4]['gl_posq'],
                                  rotmat=self.jlc.jnts[4]['gl_rotmatq'],
                                  homeconf=np.zeros(0),
                                  name='logo_02')

        self.logo_02.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "logo_02.stl"))
        self.logo_02.lnks[0]['rgba'] = [1, 0, 0, 1]
        self.logo_02.reinitialize()

        # collision checker
        if enable_cc:
            super().enable_cc()
            # cd meshes collection for precise collision checking
            self.cdmesh_collection = mc.ModelCollection()
        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "gofa5_arm.urdf")
            self._ik_solver = TracIK("world", f"gofa5_arm_link_6", urdf)
        else:
            self._ik_solver = None

    def ik(self,
           tgt_pos=np.array([.7, 0, .7]),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=500,
           tcp_jntid=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        """
        执行逆运动学 (IK) 计算,求解达到目标位置和姿态所需的关节角度
        如果存在内部 IK 求解器,则使用该求解器；否则,调用父类的 IK 方法

        :param tgt_pos: 目标位置 (3D 向量).默认为 np.array([.7, 0, .7])
        :param tgt_rotmat: 目标旋转矩阵 (3x3 矩阵).默认为单位矩阵
        :param seed_jnt_values: 关节角度的种子值,用于引导 IK 求解器.如果为 None,则使用机器人的归位姿态 (self.homeconf) 作为种子值
        :param max_niter: IK 求解器的最大迭代次数.默认为 500
        :param tcp_jntid: 工具中心点 (TCP) 所在的关节 ID
        :param tcp_loc_pos: TCP 相对于 `tcp_jntid` 关节的局部位置
        :param tcp_loc_rotmat: TCP 相对于 `tcp_jntid` 关节的局部旋转矩阵
        :param local_minima: 指定如何处理局部最小值.默认为 "accept"
        :param toggle_debug: 是否启用调试模式.默认为 False
        :return: IK 求解器计算得到的关节角度

        """
        if self._ik_solver is None:
            return super().ik(tgt_pos, tgt_rotmat, seed_jnt_values, max_niter, tcp_jntid,
                              tcp_loc_pos, tcp_loc_rotmat, local_minima, toggle_debug)
        else:
            seed_jnt_values = self.homeconf if seed_jnt_values is None else seed_jnt_values.copy()
            ik_solution = self._ik_solver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values)
            # print("ik solution is", ik_solution, tgt_pos, tgt_rotmat)
            return ik_solution

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)

        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)

    def set_tracik_joint_limits(self, lower_bounds, upper_bounds):
        if is_trac_ik:
            self._ik_solver.joint_limits = [lower_bounds, upper_bounds]
        else:
            raise Exception("Trac IK is not Correctly installed")

    def get_tracik_joint_limits(self):
        if is_trac_ik:
            return self._ik_solver.joint_limits
        else:
            raise Exception("Trac IK is not Correctly installed")


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    # # gofa5_arm.urdf文件生成代码
    # base = wd.World(cam_pos=[3.7, -4, 1.7], lookat_pos=[1.5, 0, .3])
    # gm.gen_frame().attach_to(base)
    # manipulator_instance = Gofa5Arm(enable_cc=True)
    # lm = np.asarray(manipulator_instance.get_jnt_ranges())
    # lb = lm[:, 0]
    # ub = lm[:, 1]
    # print(lm, lb, ub)
    # manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    # for cm in manipulator_meshmodel.cm_list:
    #     cm.set_rgba([*cm.get_rgba()[:3], .3])
    #     cm.attach_to(base)
    #
    # manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    # manipulator_instance.show_cdprimit()
    #
    # with open("gofa5_arm.urdf", "w") as f:
    #     f.write(str(manipulator_instance.gen_urdf()))
    # base.run()

    base = wd.World(cam_pos=[3.7, -4, 1.7], lookat_pos=[1.5, 0, .3])
    gm.gen_frame().attach_to(base)
    manipulator_instance = Gofa5Arm(enable_cc=True)

    jnt_values = manipulator_instance.get_jnt_values()
    manipulator_instance.fk(jnt_values)
    tcp_pos, tcp_rotmat = manipulator_instance.get_gl_tcp()
    st = time.time()
    tgt_pos = np.array([.65, .145, 1])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    print("tgt_pos, tgt_rotmat:", tgt_pos, tgt_rotmat)

    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, length=.3).attach_to(base)
    # base.run()
    jnt_values = manipulator_instance.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    print("ik time consumption", time.time() - st)
    print(repr(jnt_values))
    manipulator_instance.fk(jnt_values)
    print("get_gl_tcp:", manipulator_instance.get_gl_tcp())

    # array([-0.02365316,  0.21125727, -0.13446526, -0.29912526, -0.08034542,
    #         0.29821687])

    # array([-0.02357243,  0.21119083, -0.13438171, -0.29824889, -0.08033792,
    #         0.29734237])

    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    for cm in manipulator_meshmodel.cm_list:
        cm.set_rgba([*cm.get_rgba()[:3], .3])
        cm.attach_to(base)
    manipulator_instance.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    base.run()
