import math
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
import basis.robot_math as rm
import robot_sim.robots.gofa5.gofa5 as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import time
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
import os
from scipy.spatial.transform import Rotation, Slerp
import robot_con.ag145.ag145 as agctrl
import robot_con.gofa_con.gofa_con as gofa_con


def grasp_pos_rot(pos, rot, length, hight):
    """
    生成2个抓取位置
    pos：柜子的位置
    rot：柜子的姿态
    length：放置补偿距离
    hight：放置补偿高度
    """
    posrotlist = []
    Rotation_matrix = rm.homomat_from_posrot(pos, rot)
    fix_1 = rm.homomat_from_posrot(np.array([-length, 0, hight]), np.dot(np.eye(3), rm.rotmat_from_axangle(
        axis=np.array([0, 1, 0]), angle=math.pi / 2)))
    new_posrot_1 = np.dot(Rotation_matrix, fix_1)
    posrotlist.append(new_posrot_1)
    gm.gen_frame(new_posrot_1[:3, 3], new_posrot_1[:3, :3]).attach_to(base)
    fix_2 = rm.homomat_from_posrot(np.array([length, 0, hight]), np.dot(np.eye(3),
                                                                        rm.rotmat_from_axangle(axis=np.array([0, 1, 0]),
                                                                                               angle=-math.pi / 2)))
    fix_3 = rm.homomat_from_posrot(np.array([0, 0, 0]),
                                   np.dot(np.eye(3), rm.rotmat_from_axangle(axis=np.array([0, 0, 1]), angle=math.pi)))
    new_posrot_2 = np.dot(Rotation_matrix, fix_2)
    new_posrot_2 = np.dot(new_posrot_2, fix_3)
    posrotlist.append(new_posrot_2)
    gm.gen_frame(new_posrot_2[:3, 3], new_posrot_2[:3, :3]).attach_to(base)
    return posrotlist


def place_pos_rot(pos, rot, length, hight):
    """
    生成放置位置姿态
    pos：柜子的位置
    rot：柜子的姿态
    length：放置补偿距离
    hight：放置补偿高度
    """
    Rotation_matrix = rm.homomat_from_posrot(pos, rot)
    fix = rm.homomat_from_posrot(np.array([length, 0, hight]), np.dot(np.eye(3),
                                                                      rm.rotmat_from_axangle(axis=np.array([0, 1, 0]),
                                                                                             angle=-math.pi / 2)))
    new_posrot = np.dot(Rotation_matrix, fix)
    new_posrot = np.dot(new_posrot, rm.homomat_from_posrot(np.array([0, 0, 0]), np.dot(np.eye(3),
                                                                                       rm.rotmat_from_axangle(
                                                                                           axis=np.array([0, 0, 1]),
                                                                                           angle=-math.pi))))
    gm.gen_frame(new_posrot[:3, 3], new_posrot[:3, :3]).attach_to(base)
    return new_posrot


def Correction_position(angle):
    """
    修正位置
    把rot修正为z轴朝下的情况
    """
    object_rot = np.dot(np.dot(np.eye(3), rm.rotmat_from_axangle(axis=np.array([0, 0, 1]), angle=angle)),
                        rm.rotmat_from_axangle(axis=np.array([1, 0, 0]), angle=math.pi))
    return object_rot


def interpolate_rotation_matrices(R1, R2, num_steps, include_endpoints=True):
    """
    在两个旋转矩阵之间等距生成中间旋转矩阵

    参数:
    R1, R2: 3x3 旋转矩阵
    num_steps: 生成的中间矩阵数量
    include_endpoints: 是否包含起始和结束矩阵

    返回:
    list: 包含所有旋转矩阵的列表
    """
    # 验证输入矩阵是否为3x3旋转矩阵
    assert R1.shape == (3, 3), "R1 必须是3x3矩阵"
    assert R2.shape == (3, 3), "R2 必须是3x3矩阵"
    assert np.allclose(np.dot(R1, R1.T), np.eye(3), atol=1e-6), "R1 不是正交矩阵"
    assert np.allclose(np.dot(R2, R2.T), np.eye(3), atol=1e-6), "R2 不是正交矩阵"
    assert np.allclose(np.linalg.det(R1), 1.0, atol=1e-6), "R1 行列式不为1"
    assert np.allclose(np.linalg.det(R2), 1.0, atol=1e-6), "R2 行列式不为1"

    # 转换为四元数
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)

    # 创建SLERP对象
    slerp = Slerp([0, 1], Rotation.concatenate([rot1, rot2]))

    # 生成等距的时间点
    if include_endpoints:
        times = np.linspace(0, 1, num_steps + 2)
    else:
        times = np.linspace(0, 1, num_steps + 2)[1:-1]

    # 插值得到旋转
    interp_rots = slerp(times)

    # 转换为旋转矩阵
    interp_matrices = interp_rots.as_matrix()

    return list(interp_matrices)


if __name__ == '__main__':
    start = time.time()
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot_s = cbt.GOFA5(enable_cc=True)
    xxxxx = robot_s.get_gl_tcp(manipulator_name='arm')
    start_conf = np.array([0, 0, 0, 0, 0, 0])
    # ag_r = agctrl.Ag145driver()
    # rbt_r = gofa_con.GoFaArmController()
    # rbt_r.move_j(start_conf)

    # 柜子生成
    rrtc_planner = rrtc.RRTConnect(robot_s)
    object_locket = cm.CollisionModel(
        os.path.join(this_dir, "meshes", "locker.STL"),
        cdprimit_type="box", expand_radius=.001)
    object_locket.set_pos(np.array([0.9, -0.5, 0.303]))
    # object_locket_rot = Correction_position(math.pi / 2)
    # object_locket.attach_to(base)
    # base.run()
    object_locket_rot = np.dot(Correction_position(-math.pi / 2),
                               rm.rotmat_from_axangle(axis=np.array([0, 0, 1]), angle=math.pi))
    object_locket.set_rotmat(object_locket_rot)
    object_locket.attach_to(base)
    object_locket_pos = object_locket.get_pos()
    object_locket_rot = object_locket.get_rotmat()
    gm.gen_frame(object_locket_pos, object_locket_rot).attach_to(base)
    place_posrot = place_pos_rot(object_locket_pos, object_locket_rot, length=0.3, hight=0.1)
    place_posrot2 = place_pos_rot(object_locket_pos, object_locket_rot, length=0.1, hight=0.1)

    object_rack_10ml_center = cm.CollisionModel(
        os.path.join(this_dir, "meshes", "rack_10ml_center.STL"),
        cdprimit_type="box", expand_radius=.001)
    object_rack_10ml_center.set_pos(np.array([0.6, -0.2, 0.111]))
    object_rack_10ml_center_rot = Correction_position(-math.pi / 2)
    object_rack_10ml_center.set_rotmat(object_rack_10ml_center_rot)
    object_rack_10ml_center_pos = object_rack_10ml_center.get_pos()
    object_rack_10ml_center_rot = object_rack_10ml_center.get_rotmat()
    gm.gen_frame(object_rack_10ml_center_pos, object_rack_10ml_center_rot).attach_to(base)
    object_rack_10ml_center.attach_to(base)
    # base.run()

    object_rack_10ml_center_2 = cm.CollisionModel(
        os.path.join(this_dir, "meshes", "rack_10ml_center.STL"),
        cdprimit_type="box", expand_radius=.001)
    object_rack_10ml_center_2.set_pos(np.array([0.6, -0.4, 0.111]))
    object_rack_10ml_center_rot_2 = Correction_position(-math.pi / 2)
    object_rack_10ml_center_2.set_rotmat(object_rack_10ml_center_rot_2)
    object_rack_10ml_center_2.attach_to(base)

    object_rack_10ml_center_3 = cm.CollisionModel(
        os.path.join(this_dir, "meshes", "rack_10ml_center.STL"),
        cdprimit_type="box", expand_radius=.001)
    object_rack_10ml_center_3.set_pos(np.array([0.8, 0.1, 0.111]))
    object_rack_10ml_center_rot_3 = Correction_position(-math.pi / 3)
    object_rack_10ml_center_3.set_rotmat(object_rack_10ml_center_rot_3)
    object_rack_10ml_center_3.attach_to(base)

    object_rack_10ml_center_pos = object_rack_10ml_center_3.get_pos()
    object_rack_10ml_center_rot = object_rack_10ml_center_3.get_rotmat()
    gm.gen_frame(object_rack_10ml_center_pos, object_rack_10ml_center_rot).attach_to(base)
    grasp_list = grasp_pos_rot(object_rack_10ml_center_pos, object_rack_10ml_center_rot, length=0.05, hight=0.01)

    obstacle_list = [object_rack_10ml_center, object_locket, object_rack_10ml_center_2, object_rack_10ml_center_3]

    all_grasp_path = []

    for i in grasp_list:
        garsp_jnt_values = robot_s.ik(component_name='arm', tgt_pos=i[:3, 3], tgt_rotmat=i[:3, :3])
        robot_s.fk(component_name='arm', jnt_values=garsp_jnt_values)
        is_collided = robot_s.is_collided(obstacle_list=obstacle_list)
        print(is_collided)
        if is_collided == False:
            break
    grasp_path = rrtc_planner.plan(component_name="arm",
                                   start_conf=start_conf,
                                   goal_conf=garsp_jnt_values,
                                   obstacle_list=obstacle_list,
                                   ext_dist=0.1,
                                   max_time=300)
    rbt_pos = robot_s.get_gl_tcp(manipulator_name='arm')[0]
    rbt_rot = robot_s.get_gl_tcp(manipulator_name='arm')[1]
    prepare_rbt_pos = rbt_pos + np.array([0.2, 0, 0.4])
    change_pos_list0 = np.linspace(start=rbt_pos, stop=prepare_rbt_pos, num=20, endpoint=True)
    grasp_path2 = []
    for i in change_pos_list0:
        grasp_path2_jnt = robot_s.ik(component_name='arm', tgt_pos=i, tgt_rotmat=rbt_rot)
        grasp_path2_jnt = np.array([grasp_path2_jnt])
        grasp_path2.extend(grasp_path2_jnt)

    # base.run()

    change_angle_list = interpolate_rotation_matrices(rbt_rot, place_posrot[:3, :3], 10, include_endpoints=True)
    grasp_path3 = []
    for i in change_angle_list:
        grasp_path3_jnt = robot_s.ik(component_name='arm', tgt_pos=prepare_rbt_pos, tgt_rotmat=i)
        grasp_path3_jnt = np.array([grasp_path3_jnt])
        grasp_path3.extend(grasp_path3_jnt)

    change_pos_list1 = np.linspace(start=prepare_rbt_pos, stop=place_posrot[:3, 3], num=20, endpoint=True)
    grasp_path4 = []
    for i in change_pos_list1:
        grasp_path4_jnt = robot_s.ik(component_name='arm', tgt_pos=i, tgt_rotmat=place_posrot[:3, :3])
        grasp_path4_jnt = np.array([grasp_path4_jnt])
        grasp_path4.extend(grasp_path4_jnt)

    change_pos_list2 = np.linspace(start=place_posrot[:3, 3], stop=place_posrot2[:3, 3], num=20, endpoint=True)
    grasp_path5 = []
    for i in change_pos_list2:
        grasp_path5_jnt = robot_s.ik(component_name='arm', tgt_pos=i, tgt_rotmat=place_posrot[:3, :3])
        grasp_path5_jnt = np.array([grasp_path5_jnt])
        grasp_path5.extend(grasp_path5_jnt)

    # ag_r.open_g()
    all_grasp_path.extend(grasp_path)
    # rbt_r.move_jntspace_path(grasp_path)
    # ag_r.close_g()

    all_grasp_path.extend(grasp_path2)
    # rbt_r.move_jntspace_path(grasp_path2)

    all_grasp_path.extend(grasp_path3)
    # rbt_r.move_jntspace_path(grasp_path3)

    all_grasp_path.extend(grasp_path4)
    # rbt_r.move_jntspace_path(grasp_path4)

    all_grasp_path.extend(grasp_path5)
    # rbt_r.move_jntspace_path(grasp_path5)
    # ag_r.open_g()

    robot_mesh = robot_s.gen_meshmodel()
    current_robot_mesh = robot_mesh
    count = 1
    reversible_counter = 1


    def update_robot_joints(jnt_values):
        """根据关节角度更新机器人状态"""
        global current_robot_mesh

        try:
            # 移除旧的机器人模型
            if current_robot_mesh is not None:
                current_robot_mesh.detach()

            # 更新机器人关节角度（前向运动学）
            print(jnt_values)
            robot_s.fk(jnt_values=jnt_values)

            # 生成新的机器人模型
            new_mesh = robot_s.gen_meshmodel()
            new_mesh.attach_to(base)
            current_robot_mesh = new_mesh

            return True

        except Exception as e:
            print(f"更新机器人时出错: {e}")
            return False


    def update_task(task):
        """定时任务：检查并处理网络数据"""
        global count, reversible_counter
        jnt_values = all_grasp_path[count]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints(jnt_values)
            if count == len(grasp_path) - 1:
                robot_s.hold(hnd_name='hnd', objcm=object_rack_10ml_center_3)
            if count == len(all_grasp_path) - 1:
                robot_s.release(hnd_name='hnd', objcm=object_rack_10ml_center_3)
        count = count + reversible_counter
        if count == len(all_grasp_path) - 1 or count == -1:
            reversible_counter = reversible_counter * -1
        if count == -1:
            count = 0
        return task.again


    try:
        taskMgr.doMethodLater(0.1, update_task, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
