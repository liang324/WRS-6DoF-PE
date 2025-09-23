# 这段代码定义了两个类:RobotInfo 和 ObjInfo,它们用于存储和管理机器人和对象的动画信息

class RobotInfo(object):

    def __init__(self):
        """
        初始化 RobotInfo 对象

        :ivar robot_s: 机器人对象实例
        :ivar robot_component_name: 机器人的组件名称
        :ivar robot_meshmodel: 机器人的网格模型
        :ivar robot_meshmodel_parameters: 生成网格模型所需的参数
        :ivar robot_path: 机器人的运动路径
        :ivar robot_path_counter: 跟踪机器人在路径中的当前位置
        """
        self.robot_s = None
        self.robot_component_name = None
        self.robot_meshmodel = None
        self.robot_meshmodel_parameters = None
        self.robot_path = None
        self.robot_path_counter = None

    @staticmethod
    def create_anime_info(robot_s,
                          robot_component_name,
                          robot_meshmodel_parameters,
                          robot_path):
        """
        创建并返回一个 RobotInfo 实例

        :param robot_s: 机器人对象实例
        :param robot_component_name: 机器人的组件名称
        :param robot_meshmodel_parameters: 生成网格模型所需的参数
        :param robot_path: 机器人的运动路径
        :return: 初始化后的 RobotInfo 对象
        """
        anime_info = RobotInfo()
        anime_info.robot_s = robot_s
        anime_info.robot_component_name = robot_component_name
        # 生成机器人的网格模型
        anime_info.robot_meshmodel = robot_s.gen_meshmodel(robot_meshmodel_parameters)
        anime_info.robot_meshmodel_parameters = robot_meshmodel_parameters
        anime_info.robot_path = robot_path
        anime_info.robot_path_counter = 0
        return anime_info


class ObjInfo(object):

    def __init__(self):
        """
        初始化 ObjInfo 对象

        :ivar obj: 对象实例
        :ivar obj_parameters: 对象的参数,例如颜色(RGBA)
        :ivar obj_path: 对象的运动路径
        :ivar obj_path_counter: 跟踪对象在路径中的当前位置
        """
        self.obj = None
        self.obj_parameters = None
        self.obj_path = None
        self.obj_path_counter = None

    @staticmethod
    def create_anime_info(obj, obj_path=None):
        """
        创建并返回一个 ObjInfo 实例

        :param obj: 对象实例
        :param obj_path: 对象的运动路径(可选)
        :return: 初始化后的 ObjInfo 对象
        """
        anime_info = ObjInfo()
        anime_info.obj = obj
        # 获取对象的颜色参数
        anime_info.obj_parameters = [obj.get_rgba()]
        if obj_path is None:
            # 如果没有提供路径,使用对象的当前位置和旋转矩阵初始化路径
            anime_info.obj_path = [[obj.get_pos(), obj.get_rotmat()]]
        else:
            anime_info.obj_path = obj_path
        anime_info.obj_path_counter = 0
        return anime_info
