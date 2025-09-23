import math
import numpy as np
import basis.robot_math as rm
import grasping.annotation.utils as gu
from scipy.spatial import cKDTree


def plan_pushing(hnd_s,
                 objcm,
                 cone_angle=math.radians(30),
                 icosphere_level=2,
                 local_rotation_interval=math.radians(22.5),
                 max_samples=100,
                 min_dist_between_sampled_contact_points=.005,
                 contact_offset=.002,
                 toggle_debug=False):
    """
    规划推送动作

    :param hnd_s: 手柄对象
    :param objcm: 对象网格
    :param cone_angle: 推送锥角
    :param icosphere_level: 采样球面细分级别
    :param local_rotation_interval: 局部旋转间隔
    :param max_samples: 最大采样点数
    :param min_dist_between_sampled_contact_points: 采样接触点之间的最小距离
    :param contact_offset: 接触点偏移量
    :param toggle_debug: 是否开启调试模式
    :return: 推送信息列表
    """
    # 从对象表面采样接触点和法线
    contact_points, contact_normals = objcm.sample_surface(nsample=max_samples,
                                                           radius=min_dist_between_sampled_contact_points / 2,
                                                           toggle_option='normals')
    push_info_list = []
    import modeling.geometric_model as gm
    for i, cpn in enumerate(zip(contact_points, contact_normals)):
        print(f"{i} of {len(contact_points)} done!")

        # 定义推送动作
        push_info_list += gu.define_pushing(hnd_s,
                                            objcm,
                                            gl_surface_pos=cpn[0] + cpn[1] * contact_offset,
                                            gl_surface_normal=cpn[1],
                                            cone_angle=cone_angle,
                                            icosphere_level=icosphere_level,
                                            local_rotation_interval=local_rotation_interval,
                                            toggle_debug=toggle_debug)
    return push_info_list


def write_pickle_file(objcm_name, push_info_list, root=None, file_name='preannotated_push.pickle', append=False):
    """
    将推送信息列表写入 pickle 文件

    :param objcm_name: 对象名称
    :param push_info_list: 推送信息列表
    :param root: 文件保存路径
    :param file_name: 文件名
    :param append: 是否追加写入
    """
    if root is None:
        root = './'
    gu.write_pickle_file(objcm_name, push_info_list, root=root, file_name=file_name, append=append)


def load_pickle_file(objcm_name, root=None, file_name='preannotated_push.pickle'):
    """
    从 pickle 文件加载推送信息列表

    :param objcm_name: 对象名称
    :param root: 文件路径
    :param file_name: 文件名
    :return: 推送信息列表
    """
    if root is None:
        root = './'
    return gu.load_pickle_file(objcm_name, root=root, file_name=file_name)


if __name__ == '__main__':
    import os
    import basis
    import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqp

    import robot_sim.end_effectors.gripper.ag145.ag145 as ag145
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    # gripper_s = rtqp.Robotiq85GelsightPusher()
    gripper_s = ag145.Ag145()
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()
    push_info_list = plan_pushing(gripper_s, objcm, cone_angle=math.radians(60),
                                  local_rotation_interval=math.radians(45), toggle_debug=False)
    for push_info in push_info_list:
        gl_push_pos, gl_push_rotmat, hnd_pos, hnd_rotmat = push_info
        gic = gripper_s.copy()
        gic.fix_to(hnd_pos, hnd_rotmat)
        gic.gen_meshmodel().attach_to(base)
    base.run()
