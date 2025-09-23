import numpy as np
import modeling.geometric_model as gm
import modeling.collision_model as cm
import modeling.model_collection as mc
import basis.robot_math as rm


class JLChainMesh(object):
    """
    用于 JntLnks 的网格生成器类
    注意:无需反复附加节点路径到渲染中,
    一旦附加,它将始终存在.更新关节角度将直接改变附加的模型.
    """

    def __init__(self, jlobject, cdprimitive_type='box', cdmesh_type='triangles'):
        """
        初始化方法
        :param jlobject: 用于生成网格的 JntLnks 对象
        :param cdprimitive_type: 碰撞模型的基本类型(默认 'box')
        :param cdmesh_type: 碰撞网格的类型(默认 'triangles')
        author: weiwei
        date: 20200331
        """
        self.jlobject = jlobject  # 保存 JntLnks 对象
        for id in range(self.jlobject.ndof + 1):
            if self.jlobject.lnks[id]['mesh_file'] is not None and self.jlobject.lnks[id]['collision_model'] is None:
                # 如果没有设置碰撞模型,则创建并初始化一个碰撞模型
                # 步骤:1. 保持网格模型为 None；2. 直接设置碰撞模型
                self.jlobject.lnks[id]['collision_model'] = cm.CollisionModel(self.jlobject.lnks[id]['mesh_file'],
                                                                              cdprimit_type=cdprimitive_type,
                                                                              cdmesh_type=cdmesh_type)
                # 设置碰撞模型的缩放比例
                self.jlobject.lnks[id]['collision_model'].set_scale(self.jlobject.lnks[id]['scale'])

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      name='robot_mesh',
                      rgba=None):
        """
        生成网格模型,并将其返回为一个集合

        :param tcp_jnt_id: 目标关节的 ID(默认为 None)
        :param tcp_loc_pos: 相对于目标关节的局部位置(默认为 None)
        :param tcp_loc_rotmat: 相对于目标关节的局部旋转矩阵(默认为 None)
        :param toggle_tcpcs: 是否添加工具坐标系(默认为 True)
        :param toggle_jntscs: 是否添加关节坐标系(默认为 False)
        :param name: 网格模型集合的名称(默认为 'robot_mesh')
        :param rgba: 网格的颜色(默认为 None)
        :return: 一个包含网格模型的模型集合
        """
        mm_collection = mc.ModelCollection(name=name)  # 创建一个新的网格模型集合
        for id in range(self.jlobject.ndof + 1):
            if self.jlobject.lnks[id]['collision_model'] is not None:
                # 获取当前关节链接的碰撞模型
                this_collisionmodel = self.jlobject.lnks[id]['collision_model'].copy()
                pos = self.jlobject.lnks[id]['gl_pos']  # 获取该关节链接的全局位置
                rotmat = self.jlobject.lnks[id]['gl_rotmat']  # 获取该关节链接的全局旋转矩阵

                # 设置碰撞模型的齐次变换矩阵
                this_collisionmodel.set_homomat(rm.homomat_from_posrot(pos, rotmat))
                # 如果提供了颜色,则使用提供的颜色,否则使用默认颜色
                this_rgba = self.jlobject.lnks[id]['rgba'] if rgba is None else rgba
                this_collisionmodel.set_rgba(this_rgba)  # 设置颜色

                # 将该碰撞模型附加到模型集合中
                this_collisionmodel.attach_to(mm_collection)
        # 如果需要,添加工具坐标系
        if toggle_tcpcs:
            self._toggle_tcpcs(mm_collection,
                               tcp_jnt_id,
                               tcp_loc_pos,
                               tcp_loc_rotmat,
                               tcpic_rgba=np.array([.5, 0, 1, 1]),  # 设置颜色为紫色
                               tcpic_thickness=.0062)  # 设置坐标系的厚度
        # 如果需要,添加关节坐标系
        if toggle_jntscs:
            alpha = 1 if rgba == None else rgba[3]  # 如果没有提供透明度,默认为 1
            self._toggle_jntcs(mm_collection,
                               jntcs_thickness=.0062,  # 设置关节坐标系的厚度
                               alpha=alpha)  # 设置透明度
        return mm_collection  # 返回模型集合

    def gen_stickmodel(self,
                       rgba=np.array([.5, 0, 0, 1]),
                       thickness=.01,
                       joint_ratio=1.62,
                       link_ratio=.62,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=True,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='robot_stick'):
        """
        为 JntLnk 对象生成棒模型

        :param rgba: 模型的颜色,默认红色(.5, 0, 0, 1)
        :param thickness: 棒的厚度,默认为 0.01
        :param joint_ratio: 关节的比例,默认为 1.62
        :param link_ratio: 链接的比例,默认为 0.62
        :param tcp_jnt_id: 目标关节 ID(默认为 None)
        :param tcp_loc_pos: 目标位置(默认为 None)
        :param tcp_loc_rotmat: 目标旋转矩阵(默认为 None)
        :param toggle_tcpcs: 是否显示工具坐标系(默认为 True)
        :param toggle_jntscs: 是否显示关节坐标系(默认为 False)
        :param toggle_connjnt: 是否显示连接关节(默认为 False)
        :param name: 返回的模型名称,默认为 'robot_stick'
        :return: 返回一个包含棒模型的模型集合
        author: weiwei
        date: 20200331, 20201006
        """
        stickmodel = mc.ModelCollection(name=name)  # 创建一个新的模型集合
        id = 0
        loopdof = self.jlobject.ndof + 1  # 计算自由度数目
        if toggle_connjnt:
            loopdof = self.jlobject.ndof + 2  # 如果需要显示连接关节,则自由度数目加 2
        while id < loopdof:
            cjid = self.jlobject.jnts[id]['child']  # 获取子关节 ID
            jgpos = self.jlobject.jnts[id]['gl_posq']  # 获取关节的全局位置
            cjgpos = self.jlobject.jnts[cjid]['gl_pos0']  # 获取子关节的全局位置
            jgmtnax = self.jlobject.jnts[id]["gl_motionax"]  # 获取关节的全局旋转轴

            # 生成连接关节的棒(矩形类型)
            gm.gen_stick(spos=jgpos, epos=cjgpos, thickness=thickness, type="rect", rgba=rgba).attach_to(stickmodel)

            # 如果是旋转关节,添加旋转棒
            if id > 0:
                if self.jlobject.jnts[id]['type'] == "revolute":
                    gm.gen_stick(spos=jgpos - jgmtnax * thickness, epos=jgpos + jgmtnax * thickness, type="rect",
                                 thickness=thickness * joint_ratio, rgba=np.array([.3, .3, .2, rgba[3]])).attach_to(
                        stickmodel)

                # 如果是平移关节,添加平移棒
                if self.jlobject.jnts[id]['type'] == "prismatic":
                    jgpos0 = self.jlobject.jnts[id]['gl_pos0']  # 获取平移关节的初始位置
                    gm.gen_stick(spos=jgpos0, epos=jgpos, type="round", thickness=thickness * joint_ratio,
                                 rgba=np.array([.2, .3, .3, rgba[3]])).attach_to(stickmodel)
            id = cjid  # 更新关节 ID
        # 如果需要,添加工具坐标系
        if toggle_tcpcs:
            self._toggle_tcpcs(stickmodel, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat,
                               tcpic_rgba=rgba + np.array([0, 0, 1, 0]), tcpic_thickness=thickness * link_ratio)
        # 如果需要,添加关节坐标系
        if toggle_jntscs:
            self._toggle_jntcs(stickmodel, jntcs_thickness=thickness * link_ratio, alpha=rgba[3])
        # 返回包含所有棒模型的集合
        return stickmodel

    def gen_endsphere(self, rgba=None, name=''):
        """
        生成一个末端球体(es)模型,用于显示末端执行器的轨迹

        :param rgba: 球体的颜色
        :param name: 球体的名称
        :return: 返回一个包含末端球体的静态几何模型
        author: weiwei
        date: 20181003madrid, 20200331
        """
        pass
        # eesphere = gm.StaticGeometricModel(name=name)
        # if rgba is not None:
        #     gm.gen_sphere(pos=self.jlobject.jnts[-1]['linkend'], radius=.025, rgba=rgba).attach_to(eesphere)
        # return gm.StaticGeometricModel(eesphere)

    def _toggle_tcpcs(self,
                      parent_model,
                      tcp_jnt_id,
                      tcp_loc_pos,
                      tcp_loc_rotmat,
                      tcpic_rgba,
                      tcpic_thickness,
                      tcpcs_thickness=None,
                      tcpcs_length=None):
        """
        绘制工具坐标系(TCP)指示器和坐标系

        :param parent_model: 将绘制坐标系的父模型
        :param tcp_jnt_id: 单个或多个关节 ID
        :param tcp_loc_pos: 工具末端的位置信息
        :param tcp_loc_rotmat: 工具末端的旋转矩阵
        :param tcpic_rgba: 工具坐标系指示器的颜色(RGBA格式)
        :param tcpic_thickness: 工具坐标系指示器的厚度
        :param tcpcs_thickness: 坐标框架的厚度
        :param tcpcs_length: 工具坐标系坐标轴的长度
        :return: 无返回值,直接在 `parent_model` 上绘制
        author: weiwei
        date: 20201125
        """
        # 如果没有给定参数,使用默认值
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlobject.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlobject.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlobject.tcp_loc_rotmat
        if tcpcs_thickness is None:
            tcpcs_thickness = tcpic_thickness
        if tcpcs_length is None:
            tcpcs_length = tcpcs_thickness * 15

        # 获取工具末端的全局位置和旋转矩阵
        tcp_gl_pos, tcp_gl_rotmat = self.jlobject.get_gl_tcp(tcp_jnt_id,
                                                             tcp_loc_pos,
                                                             tcp_loc_rotmat)
        # 如果 tcp_jnt_id 是多个 ID,则分别为每个 ID 绘制
        if isinstance(tcp_gl_pos, list):
            for i, jid in enumerate(tcp_jnt_id):
                jgpos = self.jlobject.jnts[jid]['gl_posq']
                gm.gen_dashstick(spos=jgpos,
                                 epos=tcp_gl_pos[i],
                                 thickness=tcpic_thickness,
                                 rgba=tcpic_rgba,
                                 type="round").attach_to(parent_model)
                gm.gen_mycframe(pos=tcp_gl_pos[i],
                                rotmat=tcp_gl_rotmat[i],
                                length=tcpcs_length,
                                thickness=tcpcs_thickness,
                                alpha=tcpic_rgba[3]).attach_to(parent_model)
        else:
            jgpos = self.jlobject.jnts[tcp_jnt_id]['gl_posq']
            gm.gen_dashstick(spos=jgpos,
                             epos=tcp_gl_pos,
                             thickness=tcpic_thickness,
                             rgba=tcpic_rgba,
                             type="round").attach_to(parent_model)
            gm.gen_mycframe(pos=tcp_gl_pos,
                            rotmat=tcp_gl_rotmat,
                            length=tcpcs_length,
                            thickness=tcpcs_thickness,
                            alpha=tcpic_rgba[3]).attach_to(parent_model)

    def _toggle_jntcs(self, parentmodel, jntcs_thickness, jntcs_length=None, alpha=1):
        """
        绘制关节坐标系

        :param parentmodel: 将绘制坐标系的父模型
        :param jntcs_thickness: 坐标系的厚度
        :param jntcs_length: 坐标系的长度(默认为厚度的15倍)
        :param alpha: 坐标系的透明度(默认为 1)
        :return: 无返回值,直接在 `parentmodel` 上绘制
        author: weiwei
        date: 20201125
        """
        if jntcs_length is None:
            jntcs_length = jntcs_thickness * 15  # 默认长度为厚度的15倍
        for id in self.jlobject.tgtjnts:  # 遍历所有目标关节
            gm.gen_dashframe(pos=self.jlobject.jnts[id]['gl_pos0'],
                             rotmat=self.jlobject.jnts[id]['gl_rotmat0'],
                             length=jntcs_length,
                             thickness=jntcs_thickness,
                             alpha=alpha).attach_to(parentmodel)
            gm.gen_frame(pos=self.jlobject.jnts[id]['gl_posq'],
                         rotmat=self.jlobject.jnts[id]['gl_rotmatq'],
                         length=jntcs_length,
                         thickness=jntcs_thickness,
                         alpha=alpha).attach_to(parentmodel)
