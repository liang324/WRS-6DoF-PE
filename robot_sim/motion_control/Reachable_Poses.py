import math
import numpy as np
import basis.robot_math as rm
import robot_sim.manipulators.gofa5_arm.gofa5_arm as rbt
import visualization.panda.world as wd
import modeling.geometric_model as gm

if __name__ == '__main__':
    base = wd.World(cam_pos=[3.7, -4, 1.7], lookat_pos=[1.5, 0, .3])
    gm.gen_frame().attach_to(base)

    # 创建 Gofa5Arm 实例
    gofa5_arm = rbt.Gofa5Arm(enable_cc=True)
    gofa5_arm.gen_meshmodel().attach_to(base)

    # 定义标定板相对于 TCP 的位置和姿态
    calib_board_loc_pos = np.array([0.1, 0, 0])  # 标定板相对于 TCP 的局部位置
    calib_board_loc_rotmat = np.eye(3)  # 标定板相对于 TCP 的局部旋转矩阵

    # 生成目标位姿
    num_poses = 20  # 目标位姿数量
    target_poses = []
    for i in range(num_poses):
        # 生成随机位置
        tgt_pos = np.array([0.5 + 0.1 * math.cos(2 * math.pi * i / num_poses),
                            0.2 * math.sin(2 * math.pi * i / num_poses),
                            0.6 + 0.1 * math.sin(4 * math.pi * i / num_poses)])

        # 生成随机旋转矩阵
        tgt_rotmat = rm.rotmat_from_axangle([0, 0, 1], 2 * math.pi * i / num_poses)

        target_poses.append((tgt_pos, tgt_rotmat))

    # 逆运动学求解和可视化
    for tgt_pos, tgt_rotmat in target_poses:
        # 计算 TCP 的目标位置和姿态
        tcp_tgt_pos = tgt_pos - calib_board_loc_pos @ tgt_rotmat.T
        tcp_tgt_rotmat = tgt_rotmat @ calib_board_loc_rotmat.T

        # 逆运动学求解
        jnt_values = gofa5_arm.ik(tgt_pos=tcp_tgt_pos, tgt_rotmat=tcp_tgt_rotmat)

        if jnt_values is not None:
            # 更新机器人模型
            gofa5_arm.fk(jnt_values)
            gofa5_arm.gen_meshmodel().attach_to(base)

            # 可视化标定板
            calib_board_pos = tgt_pos
            calib_board_rotmat = tgt_rotmat
            gm.gen_frame(pos=calib_board_pos, rotmat=calib_board_rotmat, length=0.2).attach_to(base)

        else:
            print(f"IK failed for pose: {tgt_pos}, {tgt_rotmat}")

    base.run()
