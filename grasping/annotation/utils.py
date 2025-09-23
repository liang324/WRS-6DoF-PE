import math
import pickle
import numpy as np
import basis.robot_math as rm
import os
import basis
import robot_sim.end_effectors.gripper.xarm_gripper.xarm_gripper as xag
import modeling.collision_model as cm
import visualization.panda.world as wd


def define_grasp(hnd_s,
                 objcm,
                 gl_jaw_center_pos,
                 gl_jaw_center_z,
                 gl_jaw_center_y,
                 jaw_width,
                 toggle_flip=True,
                 toggle_debug=False):
    """
    定义抓取姿势

    :param hnd_s: 手部模型
    :param objcm: 目标对象的几何模型
    :param gl_jaw_center_pos: 夹爪中心位置
    :param gl_jaw_center_z: 手部接近方向
    :param gl_jaw_center_y: 拇指接触面的法线方向
    :param jaw_width: 夹爪宽度
    :param toggle_flip: 是否翻转夹爪
    :param toggle_debug: 是否开启调试模式
    :return: 抓取信息列表 [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]

    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []
    collided_grasp_info_list = []
    grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z, gl_jaw_center_y, jaw_width)
    if not hnd_s.is_mesh_collided([objcm]):
        grasp_info_list.append(grasp_info)
    else:
        collided_grasp_info_list.append(grasp_info)

    # 如果 toggle_flip 为 True,会尝试将手爪的方向进行翻转,
    # 即反转大拇指接触面的法线方向 gl_jaw_center_y,然后再次计算并检查是否发生碰撞,
    # 计算并判断碰撞后,再次将抓取信息添加到对应的列表中,

    if toggle_flip:
        grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z, -gl_jaw_center_y, jaw_width)
        if not hnd_s.is_mesh_collided([objcm]):
            grasp_info_list.append(grasp_info)
        else:
            collided_grasp_info_list.append(grasp_info)

    # 如果 toggle_debug 为 True,调试信息会显示所有发生碰撞和未发生碰撞的抓取姿势,
    # 使用 hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3])
    # 创建并附加碰撞的抓取模型,使用绿色 (rgba=[0, 1, 0, .3]) 显示未碰撞的抓取模型,

    if toggle_debug:
        for grasp_info in collided_grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for grasp_info in grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.jaw_to(jaw_width)
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    return grasp_info_list


def define_grasp_with_rotation(hnd_s,
                               objcm,
                               gl_jaw_center_pos,
                               gl_jaw_center_z,
                               gl_jaw_center_y,
                               jaw_width,
                               gl_rotation_ax,
                               rotation_interval=math.radians(120),
                               rotation_range=(math.radians(-80), math.radians(80)),
                               # rotation_interval=math.radians(60),
                               # rotation_range=(math.radians(-180), math.radians(180)),
                               toggle_flip=True,
                               toggle_debug=False):
    """
    生成一系列带有旋转变化的抓取姿势
    通过围绕指定的旋转轴旋转手爪,并检查是否与物体发生碰撞,来生成可行的抓取姿势

    :param hnd_s: 代表手爪(gripper)的对象
    :param objcm: 物体的碰撞模型,用于检测手爪与物体的碰撞
    :param gl_jaw_center_pos: 手爪的中心位置(全局坐标系)
    :param gl_jaw_center_z: 手爪的接近方向(z 轴,全局坐标系)
    :param gl_jaw_center_y: 手爪的大拇指接触面的法线方向(y 轴,全局坐标系)
    :param jaw_width: 手爪张开的宽度
    :param gl_rotation_ax: 旋转轴(全局坐标系),手爪将围绕这个轴旋转
    :param rotation_interval: 旋转角度的间隔(弧度制),控制抓取方向的变化
    :param rotation_range: 旋转角度的范围(弧度制),确定抓取方向旋转的最小和最大角度
    :param toggle_flip: 是否进行翻转抓取的标志,如果为 True,则还会生成翻转后的抓取姿势
    :param toggle_debug: 是否启用调试模式的标志,如果为 True,则会将生成的抓取姿势可视化
    :return: 一个列表,包含所有未发生碰撞的抓取姿势,每个抓取姿势都是一个列表
    包含 [jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat]

    author: chenhao, revised by weiwei
    date: 20200104
    """
    grasp_info_list = []  # 存储未发生碰撞的抓取姿势
    collided_grasp_info_list = []  # 存储发生碰撞的抓取姿势

    # 循环生成旋转抓取姿势
    for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
        tmp_rotmat = rm.rotmat_from_axangle(gl_rotation_ax, rotate_angle)  # 生成旋转矩阵
        gl_jaw_center_z_rotated = np.dot(tmp_rotmat, gl_jaw_center_z)  # 旋转手爪的接近方向
        gl_jaw_center_y_rotated = np.dot(tmp_rotmat, gl_jaw_center_y)  # 旋转手爪的大拇指接触面的法线方向
        grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z_rotated, gl_jaw_center_y_rotated,
                                             jaw_width)  # 生成抓取姿势
        # 检查是否发生碰撞
        if not hnd_s.is_mesh_collided([objcm]):
            grasp_info_list.append(grasp_info)  # 如果未发生碰撞,则添加到未碰撞列表中
        else:
            collided_grasp_info_list.append(grasp_info)  # 如果发生碰撞,则添加到碰撞列表中

    # 翻转抓取 (可选)
    if toggle_flip:
        for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
            tmp_rotmat = rm.rotmat_from_axangle(gl_rotation_ax, rotate_angle)  # 生成旋转矩阵
            gl_jaw_center_z_rotated = np.dot(tmp_rotmat, gl_jaw_center_z)  # 旋转手爪的接近方向
            gl_jaw_center_y_rotated = np.dot(tmp_rotmat, -gl_jaw_center_y)  # 旋转手爪的大拇指接触面的法线方向 (注意取反)
            grasp_info = hnd_s.grip_at_with_jczy(gl_jaw_center_pos, gl_jaw_center_z_rotated, gl_jaw_center_y_rotated,
                                                 jaw_width)  # 生成抓取姿势

            # 检查是否发生碰撞
            if not hnd_s.is_mesh_collided([objcm]):
                grasp_info_list.append(grasp_info)
            else:
                collided_grasp_info_list.append(grasp_info)

    # 调试模式 (可选)
    if toggle_debug:
        # 可视化发生碰撞的抓取姿势 (红色)
        for grasp_info in collided_grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)  # 将手爪固定到指定位置和姿态
            hnd_s.jaw_to(jaw_width)  # 设置手爪的张开宽度
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)  # 生成红色网格模型并添加到场景中

        # 可视化未发生碰撞的抓取姿势 (绿色)
        for grasp_info in grasp_info_list:
            jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)  # 将手爪固定到指定位置和姿态
            hnd_s.jaw_to(jaw_width)  # 设置手爪的张开宽度
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)  # 生成绿色网格模型并添加到场景中
    return grasp_info_list


def define_pushing(hnd_s,
                   objcm,
                   gl_surface_pos,
                   gl_surface_normal,
                   cone_angle=math.radians(30),
                   icosphere_level=2,
                   local_rotation_interval=math.radians(45),
                   toggle_debug=False):
    """
    定义推送姿势

    :param hnd_s: 手部模型
    :param objcm: 目标对象的几何模型
    :param gl_surface_pos: 用作圆锥的顶点位置
    :param gl_surface_normal: 用作圆锥的主轴方向
    :param cone_angle: 推送姿势将在此圆锥内随机化
    :param icosphere_level: 用于生成旋转矩阵的层级
    :param local_rotation_interval: 每个推送姿势的局部轴旋转间隔
    :param toggle_debug: 是否开启调试模式
    :return: 推送信息列表

    author: weiwei
    date: 20220308
    """
    push_info_list = []
    collided_push_info_list = []
    pushing_icorotmats = rm.gen_icorotmats(icolevel=icosphere_level,
                                           crop_angle=cone_angle,
                                           crop_normal=gl_surface_normal,
                                           rotation_interval=local_rotation_interval)
    for pushing_rotmat in pushing_icorotmats:
        push_info = hnd_s.push_at(gl_push_pos=gl_surface_pos, gl_push_rotmat=pushing_rotmat)
        if not hnd_s.is_mesh_collided([objcm]):
            push_info_list.append(push_info)
        else:
            collided_push_info_list.append(push_info)
    if toggle_debug:
        for push_info in collided_push_info_list:
            gl_tip_pos, gl_tip_rotmat, hnd_pos, hnd_rotmat = push_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for push_info in push_info_list:
            gl_tip_pos, gl_tip_rotmat, hnd_pos, hnd_rotmat = push_info
            hnd_s.fix_to(hnd_pos, hnd_rotmat)
            hnd_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
        base.run()
    return push_info_list


def write_pickle_file(objcm_name, grasp_info_list, root=None, file_name='preannotated_grasps.pickle', append=False):
    """
    将抓取信息列表写入 Pickle 文件

    :param objcm_name: 对象名称
    :param grasp_info_list: 抓取信息列表
    :param root: 文件存储目录
    :param file_name: 文件名
    :param append: 是否追加到现有数据
    :return: None

    author: chenhao, revised by weiwei
    date: 20200104
    """
    if root is None:
        directory = "./"
    else:
        directory = root + "/"
    try:
        data = pickle.load(open(directory + file_name, 'rb'))  # pickle 是 Python 的一个内置模块,用于序列化和反序列化 Python 对象,
        # 序列化是将对象转换为字节流的过程,以便可以将其存储在文件中或通过网络传输；反序列化则是将字节流转换回原始对象的
    except:
        print("load failed, create new data.")
        data = {}
    if append:
        data[objcm_name].extend(grasp_info_list)
    else:
        data[objcm_name] = grasp_info_list
    for k, v in data.items():
        print(k, len(v))  # 对于每个键值对,打印键 k 和其对应值 v 的长度(len(v))
    with open(directory + file_name, 'wb') as file:
        pickle.dump(data, file)
    # pickle.dump(data, open(directory + file_name, 'wb'))


# def load_pickle_file(objcm_name, root=None, file_name='preannotated_grasps.pickle'):
#     """
#     :param objcm_name:
#     :param root:
#     :param file_name:
#     :return:
#     author: chenhao, revised by weiwei
#     date: 20200105
#     """
#     if root is None:
#         directory = "./"
#     else:
#         directory = root + "/"
#     try:
#         data = pickle.load(open(directory + file_name, 'rb'))
#         for k, v in data.items():
#             print(k, len(v))
#         grasp_info_list = data[objcm_name]
#         return grasp_info_list
#     except:
#         raise ValueError("File or data not found!")


def load_pickle_file(objcm_name, root=None, file_name='preannotated_grasps.pickle'):
    """
    加载包含特定对象抓取信息的 Pickle 文件

    :param objcm_name: 要检索抓取信息的对象名称
    :param root: Pickle 文件存储的目录
    :param file_name: Pickle 文件名
    :return: 指定对象的抓取信息列表
    :raises ValueError: 如果文件缺失、对象未找到或数据损坏
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
            raise ValueError(f"Object '{objcm_name}' 未在 Pickle 文件中找到！")

        return grasp_info_list

    except pickle.UnpicklingError:
        raise ValueError("加载 Pickle 文件时出错(文件损坏或格式不兼容).")


if __name__ == '__main__':
    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = xag.XArmGripper(enable_cc=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()
    grasp_info_list = define_grasp_with_rotation(gripper_s,
                                                 objcm,
                                                 gl_jaw_center_pos=np.array([0, 0, 0]),
                                                 gl_jaw_center_z=np.array([1, 0, 0]),
                                                 gl_jaw_center_y=np.array([0, 1, 0]),
                                                 jaw_width=.04,
                                                 gl_rotation_ax=np.array([0, 0, 1]))
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gic = gripper_s.copy()
        gic.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        gic.gen_meshmodel().attach_to(base)
    base.run()
