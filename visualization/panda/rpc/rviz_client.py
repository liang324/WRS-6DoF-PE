# 这段代码定义了一个名为 RVizClient 的类,用于与远程服务器进行通信,通过 gRPC 调用远程过程

import re
import pickle
import grpc
import random
import numpy as np
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
import modeling.geometric_model as gm
import modeling.model_collection as mc
import robot_sim.robots.robot_interface as ri


class RVizClient(object):

    def __init__(self, host="localhost:18300"):
        """
        初始化 RViz 客户端

        :param host: 服务器地址,默认是 "localhost:18300"
        """
        channel = grpc.insecure_channel(host)
        self.stub = rv_rpc.RVizStub(channel)
        # self.rmt_mesh_list = [] # TODO move to server side

    def _gen_random_name(self, prefix):
        """
        生成随机名称

        :param prefix: 名称前缀
        :return: 带有前缀的随机名称
        """
        return prefix + str(random.randint(100000, 1e6))  # 6 digits

    def reset(self):
        """
        重置远程服务器上的对象和机器人状态
        """
        # code = "base.clear_internal_update_obj()\n"
        # code += "base.clear_internal_update_robot()\n"
        code = "base.clear_external_update_obj()\n"
        code += "base.clear_external_update_robot()\n"
        code += "base.clear_noupdate_model()\n"
        # code += "for item in [%s]:\n" % ', '.join(self.rmt_mesh_list)
        # code += "    item.detach()"
        # self.rmt_mesh_list = []
        self.run_code(code)

    def run_code(self, code):
        """
        在远程服务器上运行代码

        :param code: 要执行的代码字符串
        :return: 无返回值,如果执行失败则抛出异常

        author: weiwei
        date: 20201229
        """
        print(code)
        code_bytes = code.encode('utf-8')
        return_val = self.stub.run_code(rv_msg.CodeRequest(code=code_bytes)).value
        if return_val == rv_msg.Status.ERROR:
            print("服务器出问题了！！再试一次!")
            raise Exception()
        else:
            return

    def load_common_definition(self, file, line_ids=None):
        """
        从文件加载公共定义代码,并在远程服务器上执行

        :param file: 包含公共定义的文件路径
        :param line_ids: 指定要加载的行号列表.如果为 None,则加载 'main' 之前的所有行
        :return:
        """
        with open(file, 'r') as cdfile:
            if line_ids is None:
                tmp_text = cdfile.read()
                main_idx = tmp_text.find('if __name__')
                self.common_definition = tmp_text[:main_idx]
            else:
                self.common_definition = ""
                tmp_text_lines = cdfile.readlines()
                for line_id in line_ids:
                    self.common_definition += tmp_text_lines[line_id - 1]
        # 在远程执行
        print(self.common_definition)
        self.run_code(self.common_definition)

    def change_campos(self, campos):
        """
        更改远程摄像机的位置

        :param campos: 摄像机的新位置,numpy 数组格式
        :return: None
        """
        code = "base.change_campos(np.array(%s))" % np.array2string(campos, separator=',')
        self.run_code(code)

    def change_lookatpos(self, lookatpos):
        """
        更改远程摄像机的观察点位置

        :param lookatpos: 观察点的新位置,numpy 数组格式
        :return: None
        """
        code = "base.change_lookatpos(np.array(%s))" % np.array2string(lookatpos, separator=',')
        self.run_code(code)

    def change_campos_and_lookatpos(self, campos, lookatpos):
        """
        同时更改远程摄像机的位置和观察点位置

        :param campos: 摄像机的新位置,numpy 数组格式
        :param lookatpos: 观察点的新位置,numpy 数组格式
        :return: None
        """
        code = ("base.change_campos_and_lookatpos(np.array(%s), np.array(%s))" %
                (np.array2string(campos, separator=','), np.array2string(lookatpos, separator=',')))
        self.run_code(code)

    def copy_to_remote(self, loc_instance, given_rmt_robot_s_name=None):
        """
        将本地实例复制到远程服务器

        :param loc_instance: 本地实例对象
        :param given_rmt_robot_s_name: 远程实例的名称,如果为 None,则生成随机名称
        :return: 远程实例的名称
        """
        if given_rmt_robot_s_name is None:
            given_rmt_robot_s_name = self._gen_random_name(prefix='rmt_robot_s_')
        if isinstance(loc_instance, ri.RobotInterface):
            loc_instance.disable_cc()
            self.stub.create_instance(rv_msg.CreateInstanceRequest(name=given_rmt_robot_s_name,
                                                                   data=pickle.dumps(loc_instance)))
            loc_instance.enable_cc()
        else:
            self.stub.create_instance(rv_msg.CreateInstanceRequest(name=given_rmt_robot_s_name,
                                                                   data=pickle.dumps(loc_instance)))
        return given_rmt_robot_s_name

    def update_remote(self, rmt_instance, loc_instance):
        """
        更新远程实例的状态以匹配本地实例

        :param rmt_instance: 远程实例的名称
        :param loc_instance: 本地实例对象
        :return: None
        """
        if isinstance(loc_instance, ri.RobotInterface):
            code = ("%s.fk(jnt_values=np.array(%s), hnd_name='all')\n" %
                    (rmt_instance, np.array2string(loc_instance.get_jntvalues(jlc_name='all'), separator=',')))
        elif isinstance(loc_instance, gm.GeometricModel):
            code = ("%s.set_pos(np.array(%s))\n" % (
                rmt_instance, np.array2string(loc_instance.get_pos(), separator=',')) +
                    "%s.set_rotmat(np.array(%s))\n" % (
                        rmt_instance, np.array2string(loc_instance.get_rotmat(), separator=',')) +
                    "%s.set_rgba([%s])\n" % (rmt_instance, ','.join(map(str, loc_instance.get_rgba()))))
        elif isinstance(loc_instance, gm.StaticGeometricModel):
            code = "%s.set_rgba([%s])\n" % (rmt_instance, ','.join(map(str, loc_instance.get_rgba())))
        else:
            raise ValueError
        self.run_code(code)

    def show_model(self, rmt_mesh):
        """
        显示远程模型

        :param rmt_mesh: 远程模型的名称
        :return: None
        """
        code = "base.attach_noupdate_model(%s)\n" % rmt_mesh
        # code = "%s.attach_to(base)\n" % rmt_mesh
        # self.rmt_mesh_list.append(rmt_mesh)
        self.run_code(code)

    def unshow_model(self, rmt_mesh):
        """
        隐藏远程模型

        :param rmt_mesh: 远程模型的名称
        :return: None
        """
        code = "base.detach_noupdate_model(%s)\n" % rmt_mesh
        # code = "%s.detach()\n" % rmt_mesh
        # self.rmt_mesh_list.remove(rmt_mesh)
        self.run_code(code)

    def showmodel_to_remote(self, loc_mesh, given_rmt_mesh_name=None):
        """
        将本地模型复制到远程并显示

        :param loc_mesh: 本地模型对象
        :param given_rmt_mesh_name: 远程模型的名称,如果为 None,则生成随机名称
        :return: 远程模型的名称
        author: weiwei
        date: 20201231
        """
        rmt_mesh = self.copy_to_remote(loc_instance=loc_mesh, given_rmt_robot_s_name=given_rmt_mesh_name)
        self.show_model(rmt_mesh)
        return rmt_mesh

    def unshowmodel_from_remote(self, rmt_mesh):
        """
        从远程隐藏模型

        :param rmt_mesh: 远程模型的名称
        :return: None
        """
        self.unshow_model(rmt_mesh)

    def add_anime_obj(self,
                      rmt_obj,
                      loc_obj,
                      loc_obj_path,
                      given_rmt_anime_objinfo_name=None):
        """
        添加动画对象信息到远程更新列表

        :param rmt_obj: 远程对象的名称
        :param loc_obj: 本地对象,可以是 CollisionModel, Static/GeometricModel, ModelCollection
        :param loc_obj_path: 本地对象的路径信息
        :param given_rmt_anime_objinfo_name: 远程动画对象信息的名称,如果为 None,则生成随机名称
        :return: 远程动画对象信息的名称

        author: weiwei
        date: 20201231
        """
        if given_rmt_anime_objinfo_name is None:
            given_rmt_anime_objinfo_name = self._gen_random_name(prefix='rmt_anime_objinfo_')
        code = "obj_path = ["
        for pose in loc_obj_path:
            pos, rotmat = pose
            code += "[np.array(%s), np.array(%s)]," % (
                np.array2string(pos, separator=','), np.array2string(rotmat, separator=','))
        code = code[:-1] + "]\n"
        code += ("%s.set_pos(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
                 "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
                 "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
                 "%s = wd.ani.ObjInfo.create_anime_info(obj=%s, obj_path=obj_path)\n" %
                 (given_rmt_anime_objinfo_name, rmt_obj))
        code += "base.attach_external_update_obj(%s)\n" % given_rmt_anime_objinfo_name
        self.run_code(code)
        return given_rmt_anime_objinfo_name

    def add_anime_robot(self,
                        rmt_robot_s,
                        loc_robot_component_name,
                        loc_robot_meshmodel_parameters,
                        loc_robot_motion_path,
                        given_rmt_anime_robotinfo_name=None):
        """
        添加动画机器人信息到远程更新列表

        :param rmt_robot_s: 远程机器人对象
        :param loc_robot_component_name: 本地机器人组件名称
        :param loc_robot_meshmodel_parameters: 本地机器人网格模型参数
        :param loc_robot_motion_path: 本地机器人运动路径
        :param given_rmt_anime_robotinfo_name: 远程动画机器人信息的名称,如果为 None,则生成随机名称
        :return: 远程动画机器人信息的名称

        author: weiwei
        date: 20201231
        """
        if given_rmt_anime_robotinfo_name is None:
            given_rmt_anime_robotinfo_name = self._gen_random_name(prefix='rmt_anime_robotinfo_')
        code = "robot_path = ["
        for pose in loc_robot_motion_path:
            code += "np.array(%s)," % np.array2string(pose, separator=',')
        code = code[:-1] + "]\n"
        code += ("%s = wd.ani.RobotInfo.create_anime_info(%s, " %
                 (given_rmt_anime_robotinfo_name, rmt_robot_s) +
                 "'%s', " % loc_robot_component_name +
                 "%s, " % loc_robot_meshmodel_parameters + "robot_path)\n")
        code += "base.attach_external_update_robot(%s)\n" % given_rmt_anime_robotinfo_name
        self.run_code(code)
        return given_rmt_anime_robotinfo_name

    def delete_anime_obj(self, rmt_anime_objinfo):
        """
        删除远程动画对象信息

        :param rmt_anime_objinfo: 远程动画对象信息的名称
        :return: None
        """
        code = "base.detach_external_update_obj(%s)\n" % rmt_anime_objinfo
        self.run_code(code)

    def delete_anime_robot(self, rmt_anime_robotinfo):
        """
        删除远程动画机器人信息

        :param rmt_anime_robotinfo: 远程动画机器人信息的名称
        :return: None
        """
        code = "base.detach_external_update_robot(%s)\n" % rmt_anime_robotinfo
        self.run_code(code)

    def add_stationary_obj(self,
                           rmt_obj,
                           loc_obj):
        """
        添加静态对象到远程

        :param rmt_obj: 远程对象的名称
        :param loc_obj: 本地对象
        :return: None
        """
        code = ("%s.set_pos(np.array(%s)\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
                "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
                "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
                "base.attach_noupdate_model(%s)\n" % (rmt_obj))
        self.run_code(code)

    def delete_stationary_obj(self, rmt_obj):
        """
        删除远程静态对象

        :param rmt_obj: 远程对象的名称
        :return: None
        """
        code = "base.delete_noupdate_model(%s)" % rmt_obj
        self.run_code(code)

    def add_stationary_robot(self,
                             rmt_robot_s,
                             loc_robot_s,
                             given_rmt_robot_meshmodel_name=None):
        """
        添加静态机器人到远程

        :param rmt_robot_s: 远程机器人对象
        :param loc_robot_s: 本地机器人对象
        :param given_rmt_robot_meshmodel_name: 远程机器人网格模型的名称,如果为 None,则生成随机名称
        :return: 在远程端创建的机器人网格模型的名称
        """
        if given_rmt_robot_meshmodel_name is None:
            given_rmt_robot_meshmodel_name = self._gen_random_name(prefix='rmt_robot_meshmodel_')
        jnt_angles_str = np.array2string(loc_robot_s.get_jnt_values(component_name='all'), separator=',')
        code = ("%s.fk(hnd_name='all', jnt_values=np.array(%s))\n" % (rmt_robot_s, jnt_angles_str) +
                "%s = %s.gen_meshmodel()\n" % (given_rmt_robot_meshmodel_name, rmt_robot_s) +
                "base.attach_noupdate_model(%s)\n" % given_rmt_robot_meshmodel_name)
        self.run_code(code)
        return given_rmt_robot_meshmodel_name

    def delete_stationary_robot(self, rmt_robot_meshmodel):
        """
        删除远程静态机器人

        :param rmt_robot_meshmodel: 远程机器人网格模型的名称
        :return: None
        """
        code = "base.delete_noupdate_model(%s)" % rmt_robot_meshmodel
        self.run_code(code)

    def remote_load_img(self):
        code = "base.set_img_texture()"
        self.run_code(code)