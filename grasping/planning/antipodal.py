import math
import numpy as np
import basis.robot_math as rm
import grasping.annotation.utils as gau
from scipy.spatial import cKDTree
import pickle
import os


def plan_contact_pairs(objcm,
                       max_samples=100,
                       min_dist_between_sampled_contact_points=.005,
                       angle_between_contact_normals=math.radians(160),
                       toggle_sampled_points=False):
    """
    使用射线投射找到接触对,由于 min_dist 约束,最终返回的接触对数量可能小于给定的 max_samples,因为受最小距离约束

    :param objcm: 目标对象的几何模型
    :param max_samples: 采样的最大数量
    :param min_dist_between_sampled_contact_points: 采样点之间的最小距离
    :param angle_between_contact_normals: 接触法线之间的角度
    :param toggle_sampled_points: 是否返回采样点
    :return: [[contact_p0, contact_p1], ...]

    author: weiwei
    date: 20190805, 20210504
    """
    # 调用 sample_surface 方法从物体表面采样接触点和法线.max_samples 指定要采样的点的数量,radius 用于控制采样的密度,toggle_option 指定是否返回法线.
    contact_points, contact_normals = objcm.sample_surface(nsample=max_samples,
                                                           radius=min_dist_between_sampled_contact_points / 2,
                                                           toggle_option='normals')
    # 初始化接触点对列表和 KD 树
    # 初始化一个空列表 contact_pairs 用于存储找到的接触点对
    contact_pairs = []
    # 使用 cKDTree 创建一个 KD 树,以便快速查找接触点之间的距离
    tree = cKDTree(contact_points)
    # near_history 数组用于记录哪些接触点已经被标记为“近邻”,以避免重复处理
    near_history = np.array([0] * len(contact_points), dtype=bool)  # 最终生成的 near_history 数组的所有元素都将是 False
    # 遍历接触点
    for i, contact_p0 in enumerate(contact_points):
        if near_history[i]:  # 遍历每个接触点 contact_p0,如果该点已经被标记为“近邻”,则跳过
            continue
        contact_n0 = contact_normals[i]  # 获取对应的法线 contact_n0

        # 射线投射 从 contact_p0 向外发射一条射线,检查是否与物体表面相交.射线的起点稍微偏离接触点,以避免与物体表面重合
        # 起点: contact_p0 - contact_n0 * .001
        # 这个表达式表示从接触点 contact_p0 向法线方向 contact_n0 反向移动 0.001 单位,作为射线的起点.这个小的偏移量是为了确保射线从接触点稍微向外发射,避免与物体表面重合.
        # 终点: contact_p0 - contact_n0 * 100
        # 这个表达式表示从接触点 contact_p0 向法线方向 contact_n0 反向移动 100 单位,作为射线的终点.这意味着射线将沿着法线方向延伸很长的距离

        # 100个单位长度的射线可能不足以覆盖物体较大的范围 暂时没解决
        # ray_length = max(objcm.get_size())  # 获取物体的最大尺寸作为射线长度
        # hit_points, hit_normals = objcm.ray_hit(contact_p0 - contact_n0 * 0.001, contact_p0 - contact_n0 * ray_length)

        hit_points, hit_normals = objcm.ray_hit(contact_p0 - contact_n0 * .001, contact_p0 - contact_n0 * 100)
        # 检查交点
        if len(hit_points) > 0:
            # 如果射线与物体表面相交,遍历所有交点 contact_p1 和对应的法线 contact_n1
            for contact_p1, contact_n1 in zip(hit_points, hit_normals):
                # 检查 contact_n0 和 contact_n1 之间的夹角是否小于指定的阈值,以确保它们是有效的接触点,也就是
                if np.dot(contact_n0, contact_n1) < -math.cos(angle_between_contact_normals):
                    # 使用 KD 树查找与 contact_p1 近邻的接触点,确保它们之间的距离大于最小距离
                    near_points_indices = tree.query_ball_point(contact_p1, min_dist_between_sampled_contact_points)
                    if len(near_points_indices):
                        for npi in near_points_indices:
                            # 如果找到近邻接触点,检查它们的法线与 contact_n1 的夹角,并标记为“近邻”
                            if np.dot(contact_normals[npi], contact_n1) > math.cos(angle_between_contact_normals):
                                near_history[npi] = True
                    # 添加接触点对  将有效的接触点对添加到 contact_pairs 列表中.
                    contact_pairs.append([[contact_p0, contact_n0], [contact_p1, contact_n1]])
    if toggle_sampled_points:  # 如果 toggle_sampled_points 为真,返回接触点对和采样的接触点；否则,仅返回接触点对.
        return contact_pairs, contact_points
    return contact_pairs


def plan_grasps(hnd_s,
                objcm,
                angle_between_contact_normals=math.radians(160),
                openning_direction='loc_x',
                rotation_interval=math.radians(22.5),
                max_samples=100,
                min_dist_between_sampled_contact_points=.005,
                contact_offset=.002):
    """
    根据给定的物体模型和夹爪类型规划抓取点

    :param hnd_s: 夹爪的实例,包含夹爪的属性和方法
    :param objcm: 物体的几何模型,用于计算接触点
    :param angle_between_contact_normals: 接触法线之间的角度,默认为160度,影响抓取策略的多样性
    :param openning_direction: 夹爪的开口方向,可以是 'loc_x' 或 'loc_y',取决于夹爪的类型
    :param rotation_interval: 夹爪在抓取时的旋转间隔,默认为22.5度
    :param max_samples: 最大样本数,用于生成接触点对
    :param min_dist_between_sampled_contact_points: 生成的接触点之间的最小距离
    :param contact_offset: 接触点的偏移量,避免夹爪与物体表面过于接近
    :return: 返回一个包含抓取信息的列表.a list [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]

    这段代码的目的是过滤掉那些夹爪宽度大于最大范围的接触点对,以确保抓取是可行的,但它并没有检查接触点是否位于物体的表面或内部
    解决试管架内部接触点的方法
    射线检测或法线检测:为了避免选择位于试管架内部的接触点,可以考虑在找到接触点后,检查这些接触点是否位于物体表面
    比如,可以沿接触点的法线方向发射一条射线,检查是否与物体表面相交.如果接触点位于物体内部,射线将不会与物体的外表面相交

    体积判断:如果您的物体模型是封闭的,可以利用一些物理引擎或几何库来判断接触点是否在物体的表面之外,而不是在物体的内部

    """
    # 计算接触点对, 调用根据物体模型生成接触点对,每个元素包含两个接触点及其法线
    contact_pairs = plan_contact_pairs(objcm,
                                       max_samples=max_samples,
                                       min_dist_between_sampled_contact_points=min_dist_between_sampled_contact_points,
                                       angle_between_contact_normals=angle_between_contact_normals)
    # 初始化一个空列表 存储生成的抓取信息
    grasp_info_list = []
    # 遍历接触点对
    for i, cp in enumerate(contact_pairs):
        print(f"{i} of {len(contact_pairs)} done!")  # 打印当前进度
        contact_p0, contact_n0 = cp[0]  # 从接触点对中提取接触点和法线
        contact_p1, contact_n1 = cp[1]  # 从接触点对中提取接触点和法线
        contact_center = (contact_p0 + contact_p1) / 2  # 计算接触点的中心位置
        jaw_width = np.linalg.norm(contact_p0 - contact_p1) + contact_offset * 2  # 计算夹爪的宽度,并考虑接触偏移量

        # 如果计算出的夹爪宽度超过夹爪的最大宽度范围,则跳过该接触点对
        if jaw_width > hnd_s.jawwidth_rng[1]:
            continue
        # 根据开口方向确定夹爪的中心轴和法线方向.使用法线计算夹爪的其他方向向量
        if openning_direction == 'loc_x':
            jaw_center_x = contact_n0
            jaw_center_z = rm.orthogonal_vector(contact_n0)
            jaw_center_y = np.cross(jaw_center_z, jaw_center_x)
        elif openning_direction == 'loc_y':
            jaw_center_y = contact_n0
            jaw_center_z = rm.orthogonal_vector(contact_n0)
        # if openning_direction == 'loc_x':
        #     jaw_center_x = contact_n0
        #     jaw_center_z = rm.orthogonal_vector(contact_n0)
        #     jaw_center_y = np.cross(jaw_center_z, jaw_center_x)
        # elif openning_direction == 'loc_y':
        #     jaw_center_y = contact_n0
        #     jaw_center_z = rm.orthogonal_vector(contact_n0)
        #     jaw_center_x = np.cross(jaw_center_y, jaw_center_z)
        else:
            # 如果开口方向不正确,则抛出错误
            raise ValueError("Openning direction must be loc_x or loc_y!")
        # 生成抓取信息,并将其添加到 grasp_info_list 中.该函数会考虑夹爪的旋转和其他参数.
        grasp_info_list += gau.define_grasp_with_rotation(hnd_s,
                                                          objcm,
                                                          gl_jaw_center_pos=contact_center,
                                                          gl_jaw_center_z=jaw_center_z,
                                                          gl_jaw_center_y=jaw_center_y,
                                                          jaw_width=jaw_width,
                                                          gl_rotation_ax=contact_n0,
                                                          rotation_interval=rotation_interval,
                                                          toggle_flip=True)
    return grasp_info_list


def write_pickle_file(objcm_name, grasp_info_list, root=None, file_name='preannotated_grasps.pickle', append=False):
    """
    将抓取信息列表写入 pickle 文件

    :param objcm_name: 对象名称
    :param grasp_info_list: 抓取信息列表
    :param root: 文件保存路径
    :param file_name: 文件名
    :param append: 是否追加写入
    """
    if root is None:
        root = './'
    gau.write_pickle_file(objcm_name, grasp_info_list, root=root, file_name=file_name, append=append)


def load_pickle_file(objcm_name, root=None, file_name='preannotated_grasps.pickle'):
    """
    加载包含特定物体抓取信息的 pickle 文件

    :param objcm_name: 要检索抓取信息的物体的名称
    :param root: pickle 文件所在的目录.如果未指定,则默认为当前目录
    :param file_name: pickle 文件的名称
    :return: 指定物体的抓取信息列表
    :raises ValueError: 如果文件缺失、未找到物体或数据损坏,则抛出异常
    """
    # 设置默认路径
    directory = os.path.join(root if root else "./")
    file_path = os.path.join(directory, file_name)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' not found!")
    try:
        # 读取 Pickle 文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # 打印所有对象名称及其抓取数据数量
        for k, v in data.items():
            print(k, len(v))
        # 获取目标物体的抓取数据
        grasp_info_list = data.get(objcm_name, None)
        if grasp_info_list is None:
            raise ValueError(f"Object '{objcm_name}' 未在 pickle 文件中找到！")
        return grasp_info_list
    except pickle.UnpicklingError:
        raise ValueError("加载 pickle 文件时出错(文件损坏或格式不兼容).")


if __name__ == '__main__':
    import os
    import basis
    import robot_sim.end_effectors.gripper.xarm_gripper.xarm_gripper as xag
    import robot_sim.end_effectors.gripper.ag145.ag145 as ag145
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    # gripper_s = xag.XArmGripper(enable_cc=True)
    gripper_s = ag145.Ag145(enable_cc=True)

    objpath = os.path.join(basis.__path__[0], 'objects', 'bowl.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()

    grasp_info_list = plan_grasps(gripper_s, objcm, min_dist_between_sampled_contact_points=.02)
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gic = gripper_s.copy()
        gic.fix_to(hnd_pos, hnd_rotmat)
        gic.jaw_to(jaw_width)
        print(hnd_pos, hnd_rotmat)

        gic.gen_meshmodel().attach_to(base)
    base.run()
