import numpy as np
import basis.robot_math as rm
import scipy.optimize as sopt
import motion.optimization_based.incremental_nik as inik
import pickle


def load_calibration_data(file="./depth_sensor_calib_mat.pkl", has_sensor_and_real_points=False):
    """
    加载校准数据

    :param file: 校准数据文件路径
    :param has_sensor_and_real_points: 是否包含传感器和实际点数据
    :return: 仿射矩阵、实际坐标点数组、传感器坐标点数组

    author: weiwei
    date: 20210519
    """
    if has_sensor_and_real_points:
        affine_mat, pos_in_real_array, pos_in_sensor_array = pickle.load(open(file, "rb"))
    else:
        affine_mat = pickle.load(open(file, "rb"))
        pos_in_real_array = None
        pos_in_sensor_array = None
    return affine_mat, pos_in_real_array, pos_in_sensor_array


class DepthCaliberator(object):
    """
    深度校准器类,用于校准机器人在传感器坐标系中的位置
    """

    def __init__(self, robot_x, robot_s):
        """
        初始化 DepthCaliberator 对象

        :param robot_x: 机器人对象,用于执行动作
        :param robot_s: 机器人对象,用于传感器坐标系校准
        """
        self.robot_x = robot_x
        self.robot_s = robot_s

    def _find_tcp_in_sensor(self, component_name, action_pos, action_rotmat, sensor_marker_handler):
        """
        在传感器坐标系中找到机器人工具中心点(TCP)的位姿

        :param component_name: 机器人组件名称
        :param action_pos: 动作中心位置
        :param action_rotmat: 动作旋转矩阵
        :param sensor_marker_handler: 传感器标记处理器
        :return: [估计的 TCP 中心在传感器中的位置,标记形成的球体半径]

        author: weiwei
        date: 20210408
        """

        def _fit_sphere(p, coords):
            x0, y0, z0, radius = p
            x, y, z = coords.T
            return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

        _err_fit_sphere = lambda p, x: _fit_sphere(p, x) - p[3]

        marker_pos_in_sensor_list = []
        rot_range_x = [np.array([1, 0, 0]), [-30, -15, 0, 15, 30]]
        rot_range_y = [np.array([0, 1, 0]), [-30, -15, 15, 30]]
        rot_range_z = [np.array([0, 0, 1]), [-90, -60, -30, 30, 60]]
        range_axes = [rot_range_x, rot_range_y, rot_range_z]
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        for axisid in range(3):
            axis = range_axes[axisid][0]
            for angle in range_axes[axisid][1]:
                goal_pos = action_pos
                goal_rotmat = np.dot(rm.rotmat_from_axangle(axis, angle), action_rotmat)
                jnt_values = self.robot_s.ik(component_name=component_name,
                                             tgt_pos=goal_pos,
                                             tgt_rotmat=goal_rotmat,
                                             seed_jnt_values=last_jnt_values)
                self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                if jnt_values is not None and not self.robot_s.is_collided():
                    last_jnt_values = jnt_values
                    self.robot_x.move_jnts(component_name, jnt_values)
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        marker_pos_in_sensor_list.append(marker_pos_in_sensor)
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        if len(marker_pos_in_sensor_list) < 3:
            return [None, None]
        center_in_camera_coords_array = np.asarray(marker_pos_in_sensor_list)
        # try:
        initial_guess = np.ones(4) * .001
        initial_guess[:3] = np.mean(center_in_camera_coords_array, axis=0)
        final_estimate, flag = sopt.leastsq(_err_fit_sphere, initial_guess, args=(center_in_camera_coords_array,))
        if len(final_estimate) == 0:
            return [None, None]
        return np.array(final_estimate[:3]), final_estimate[3]

    def find_board_center_in_hand(self,
                                  component_name,
                                  sensor_marker_handler,
                                  action_center_pos=np.array([.3, -.05, .2]),
                                  action_center_rotmat=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
                                  action_dist=.1):
        """
        找到手中板子的中心位置

        :param component_name: 机器人组件名称
        :param sensor_marker_handler: 传感器标记处理器,用于获取标记中心
        :param action_center_pos: 动作中心位置
        :param action_center_rotmat: 动作中心旋转矩阵
        :param action_dist: 动作距离,用于移动
        :return: 标记在手中的位置

        author: weiwei
        date: 20210408, 20210519
        """
        # 找到工具中心点在传感器坐标系中的位置
        tcp_in_sensor, radius_by_markers = self._find_tcp_in_sensor(component_name=component_name,
                                                                    action_pos=action_center_pos,
                                                                    action_rotmat=action_center_rotmat,
                                                                    aruco_info=aruco_info)
        # 备份当前关节值
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # 移动到动作中心位置和旋转矩阵
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_pos,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center is not reachable. Try a different pos or robtmat!")

        # 移动到 x+action_dist
        action_center_dist_x = action_center_pos + action_center_rotmat[:, 0] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_x,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_xplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with xplus is not reachable. Try a different pos or robtmat!")

        # 移动到 y+action_dist
        action_center_dist_y = action_center_pos + action_center_rotmat[:, 1] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_y,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_yplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with yplus is not reachable. Try a different pos or robtmat!")

        # 移动到 z+action_dist
        action_center_dist_z = action_center_pos + action_center_rotmat[:, 2] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_z,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_zplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with zplus is not reachable. Try a different pos or robtmat!")

        # 计算标记在传感器坐标系中的旋转矩阵
        unnormalized_marker_mat_in_sensor = np.array([marker_pos_xplus_in_sensor - marker_pos_in_sensor,
                                                      marker_pos_yplus_in_sensor - marker_pos_in_sensor,
                                                      marker_pos_zplus_in_sensor - marker_pos_in_sensor]).T
        marker_rotmat_in_sensor, r = np.linalg.qr(unnormalized_marker_mat_in_sensor)

        # 计算标记在手中的位置
        marker_pos_in_hnd = np.dot(marker_rotmat_in_sensor.T, marker_pos_in_sensor - tcp_in_sensor)

        # 恢复机器人到初始关节值
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return marker_pos_in_hnd

    def calibrate(self,
                  component_name,
                  sensor_marker_handler,
                  marker_pos_in_hnd=None,
                  action_pos_list=[np.array(.3, -.2, .9), np.array(.3, .2, .9),
                                   np.array(.4, -.2, .9), np.array(.4, .2, .9),
                                   np.array(.3, -.2, 1.1), np.array(.3, .2, 1.1),
                                   np.array(.4, -.2, 1.1), np.array(.4, .2, 1.1)],
                  action_rotmat_list=[np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T] * 8,
                  save_calib_file='depth_sensor_calib_mat.pkl',
                  save_sensor_and_real_points=False):
        """
        校准传感器与机器人之间的关系

        :param component_name: 机器人组件名称
        :param sensor_marker_handler: 传感器标记处理器,用于获取标记中心
        :param marker_pos_in_hnd: 标记在手中的位置
        :param action_pos_list: 动作位置列表
        :param action_rotmat_list: 动作旋转矩阵列表
        :param save_calib_file: 保存校准数据的文件名
        :param save_sensor_and_real_points: 是否保存传感器和实际点
        :return: 仿射矩阵

        author: weiwei
        date: 20191228
        """
        # 如果没有提供标记在手中的位置,调用 find_board_center_in_hand 方法获取
        if marker_pos_in_hnd is None:
            marker_pos_in_hnd = self.find_board_center_in_hand(component_name=component_name,
                                                               sensor_marker_handler=sensor_marker_handler,
                                                               action_center_pos=action_pos_list,
                                                               action_center_rotmat=action_rotmat_list,
                                                               action_dist=action_dist)

        pos_in_real_list = []  # 用于存储实际坐标系中的位置
        pos_in_sensor_list = []  # 用于存储传感器坐标系中的位置
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)  # 备份当前关节值
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()  # 获取左臂手的关节值

        # 遍历动作位置列表,进行校准
        for i, action_pos in enumerate(action_pos_list):
            jnt_values = self.robot_s.ik(component_name=component_name,
                                         tgt_pos=action_pos,
                                         tgt_rotmat=action_rotmat_list[i],
                                         seed_jnt_values=last_jnt_values)
            if jnt_values is not None:
                self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                last_jnt_values = jnt_values
                if not self.robot_s.is_collided():
                    self.robot_x.move_jnts(component_name, jnt_values)
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        pos_in_real_list.append(action_pos + np.dot(action_rotmat_list[i], marker_pos_in_hnd))
                        pos_in_sensor_list.append(marker_pos_in_sensor)
                else:
                    print(f"The {i}th action pose is collided!")
            else:
                print(f"The {i}th action pose is reachable!")

        # 恢复机器人到初始关节值
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)

        # 计算仿射矩阵
        pos_in_real_array = np.array(pos_in_real_list)
        pos_in_sensor_array = np.array(pos_in_sensor_list)
        affine_mat = rm.affine_matrix_from_points(pos_in_sensor_array.T, pos_in_real_array.T)

        # 保存校准数据
        if save_sensor_and_real_points:
            data = [affine_mat, pos_in_real_array, pos_in_sensor_array]
        else:
            data = affine_mat
        pickle.dump(data, open('./' + save_calib_file, "wb"))
        return affine_mat

    def refine_with_template(self, affine_mat, template_file):
        """
        通过与模板匹配来优化仿射矩阵

        :param affine_mat: 初始仿射矩阵
        :param template_file: 模板文件
        :return: 优化后的仿射矩阵

        author: weiwei
        date: 20191228, 20210519
        """
        pass
