import numpy as np
import modeling.geometric_model as gm



class JLTreeMesh(object):
    """
    JLTree 的网格生成类
    注意:不需要重复地将节点路径附加到渲染器,一旦附加,它将一直存在.更新关节角度时,将直接更改附加的模型.
    """

    def __init__(self, jltree_obj):
        """
        初始化 JLTreeMesh 类
        :param jltree_obj: 传入一个 JLTree 对象
        author: weiwei
        date: 20200331
        """
        self.jltree_obj = jltree_obj  # 将传入的 JLTree 对象赋值给实例变量

    def gen_stickmodel(self, rgba=np.array([.5, 0, 0, 1]), thickness=.01, jointratio=1.62, linkratio=.62,
                       tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None,
                       toggletcpcs=True, togglejntscs=False, togglecntjnt=False, name='robotstick'):
        """
        为 self.jltree_obj 生成一个 stick 模型(棒状模型)

        :param rgba: 颜色,默认红色
        :param thickness: 棒的厚度,默认值为 0.01
        :param jointratio: 关节比例,默认值为 1.62
        :param linkratio: 链节比例,默认值为 0.62
        :param tcp_jntid: 工具中心点对应的关节 ID,默认值为 None
        :param tcp_localpos: 工具中心点的局部位置,默认值为 None
        :param tcp_localrotmat: 工具中心点的局部旋转矩阵,默认值为 None
        :param toggletcpcs: 是否显示工具坐标系,默认显示(True)
        :param togglejntscs: 是否显示关节坐标系,默认不显示(False)
        :param togglecntjnt: 是否显示连接关节,默认不显示(False)
        :param name: 模型名称,默认值为 'robotstick'
        :return: 生成的 stick 模型
        author: weiwei
        date: 20200331, 20201006, 20201205
        """
        stickmodel = gm.StaticGeometricModel(name=name)  # 创建一个静态几何模型
        id = 0
        loopdof = self.jlobject.ndof + 1  # 根据自由度(ndof)确定循环次数
        if togglecntjnt:
            loopdof = self.jlobject.ndof + 2  # 如果显示连接关节,则循环次数增加

        # 遍历每个关节并生成模型
        while id < loopdof:
            cjid = self.jlobject.joints[id]['child']  # 获取当前关节的子关节
            jgpos = self.jlobject.joints[id]['g_posq']  # 获取关节的全局位置
            cjgpos = self.jlobject.joints[cjid]['g_pos0']  # 获取子关节的全局位置
            jgmtnax = self.jlobject.joints[id]["g_mtnax"]  # 获取关节的全局旋转轴

            # 生成棒状模型连接当前关节和子关节
            gm.gen_stick(spos=jgpos, epos=cjgpos, thickness=thickness, type="rect", rgba=rgba).attach_to(stickmodel)

            # 处理转动关节(revolute)和滑动关节(prismatic)
            if id > 0:
                if self.jlobject.joints[id]['type'] == "revolute":
                    gm.gen_stick(spos=jgpos - jgmtnax * thickness, epos=jgpos + jgmtnax * thickness, type="rect",
                                 thickness=thickness * jointratio, rgba=np.array([.3, .3, .2, 1])).attach_to(stickmodel)
                if self.jlobject.joints[id]['type'] == "prismatic":
                    jgpos0 = self.jlobject.joints[id]['g_pos0']  # 获取滑动关节的初始位置
                    gm.gen_stick(spos=jgpos0, epos=jgpos, type="round", hickness=thickness * jointratio,
                                 rgba=np.array([.2, .3, .3, 1])).attach_to(stickmodel)
            id = cjid  # 更新当前关节为子关节

        # 绘制工具坐标系(TCP)
        if toggletcpcs:
            self._toggle_tcpcs(stickmodel, tcp_jntid, tcp_localpos, tcp_localrotmat,
                               tcpic_rgba=rgba + np.array([0, 0, 1, 0]), tcpic_thickness=thickness * linkratio)
        # 绘制所有关节坐标系
        if togglejntscs:
            self._toggle_jntcs(stickmodel, jntcs_thickness=thickness * linkratio)
        return stickmodel

    def gen_endsphere(self, rgba=None, name=''):
        """
        生成一个末端球体,用于显示末端执行器的轨迹

        :param rgba: 末端球体的颜色,默认为 None
        :param name: 末端球体的名称,默认为空字符串
        :return: 返回生成的静态几何模型(末端球体)
        author: weiwei
        date: 20181003madrid, 20200331
        """
        eesphere = gm.StaticGeometricModel(name=name)  # 创建一个静态几何模型,表示末端球体
        if rgba is not None:
            # 如果提供了颜色,生成一个球体并附加到模型中
            gm.gen_sphere(pos=self.jlobject.joints[-1]['linkend'], radius=.025, rgba=rgba).attach_to(eesphere)
        return gm.StaticGeometricModel(eesphere)

    def _toggle_tcpcs(self, parentmodel, tcp_jntid, tcp_localpos, tcp_localrotmat, tcpic_rgba, tcpic_thickness,
                      tcpcs_thickness=None, tcpcs_length=None):
        """
        绘制工具坐标系 (TCP) 和工具指示器,显示末端执行器的坐标系

        :param parentmodel: 用于绘制坐标系的父模型
        :param tcp_jntid: 末端执行器关节 ID(可以是单个 ID 或 ID 列表)
        :param tcp_localpos: 工具的局部位置
        :param tcp_localrotmat: 工具的局部旋转矩阵
        :param tcpic_rgba: 工具指示器的颜色
        :param tcpic_thickness: 工具指示器的厚度
        :param tcpcs_thickness: TCP 坐标系的厚度
        :param tcpcs_length: TCP 坐标系的长度
        :return: None
        author: weiwei
        date: 20201125
        """
        if tcp_jntid is None:
            tcp_jntid = self.jlobject.tcp_jntid
        if tcp_localpos is None:
            tcp_localpos = self.jlobject.tcp_localpos
        if tcp_localrotmat is None:
            tcp_localrotmat = self.jlobject.tcp_localrotmat
        if tcpcs_thickness is None:
            tcpcs_thickness = tcpic_thickness  # 如果没有指定 tcpcs_thickness,使用 tcpic_thickness
        if tcpcs_length is None:
            tcpcs_length = tcpcs_thickness * 15  # 如果没有指定 tcpcs_length,默认为 tcpcs_thickness 的 15 倍

        # 获取全局 TCP 位置和旋转矩阵
        tcp_globalpos, tcp_globalrotmat = self.jlobject.get_gl_tcp(tcp_jntid, tcp_localpos, tcp_localrotmat)
        if isinstance(tcp_globalpos, list):
            # 如果 tcp_globalpos 是列表,表示多个 TCP,分别绘制每个 TCP 坐标系
            for i, jid in enumerate(tcp_jntid):
                jgpos = self.jlobject.joints[jid]['g_posq']  # 获取当前关节的全局位置
                gm.gen_dumbbell(spos=jgpos, epos=tcp_globalpos[i], thickness=tcpic_thickness,
                                rgba=tcpic_rgba).attach_to(parentmodel)
                gm.gen_frame(pos=tcp_globalpos[i], rotmat=tcp_globalrotmat[i], length=tcpcs_length,
                             thickness=tcpcs_thickness, alpha=1).attach_to(parentmodel)
        else:
            # 如果只有一个 TCP,绘制单个坐标系
            jgpos = self.jlobject.joints[tcp_jntid]['g_posq']
            gm.gen_dumbbell(spos=jgpos, epos=tcp_globalpos, thickness=tcpic_thickness, rgba=tcpic_rgba).attach_to(
                parentmodel)
            gm.gen_frame(pos=tcp_globalpos, rotmat=tcp_globalrotmat, length=tcpcs_length, thickness=tcpcs_thickness,
                         alpha=1).attach_to(parentmodel)

    def _toggle_jntcs(self, parentmodel, jntcs_thickness, jntcs_length=None):
        """
        绘制关节坐标系(Joint Coordinate Systems,JntCS)

        :param parentmodel: 绘制坐标系的父模型,表示将要显示坐标系的地方
        :param jntcs_thickness: 坐标系的厚度
        :param jntcs_length: 坐标系的长度(默认为厚度的 15 倍)
        :return: None
        author: weiwei
        date: 20201125
        """
        if jntcs_length is None:
            # 如果没有指定长度,默认使用厚度的 15 倍作为坐标系的长度
            jntcs_length = jntcs_thickness * 15
        for id in self.jlobject.tgtjnts:
            # 对于每个目标关节,绘制两种坐标系
            gm.gen_dashframe(pos=self.jlobject.joints[id]['g_pos0'], rotmat=self.jlobject.joints[id]['g_rotmat0'],
                             length=jntcs_length, thickness=jntcs_thickness).attach_to(parentmodel)
            gm.gen_frame(pos=self.jlobject.joints[id]['g_posq'], rotmat=self.jlobject.joints[id]['g_rotmatq'],
                         length=jntcs_length, thickness=jntcs_thickness, alpha=1).attach_to(parentmodel)
