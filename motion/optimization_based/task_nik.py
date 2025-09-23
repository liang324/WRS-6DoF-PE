import math
import numpy as np
import basis.robot_math as rm
import warnings as wns


class NIK(object):
    def __init__(self, robot, component_name, wln_ratio=.05):
        """
        初始化 NIK 类的实例

        :param robot: 机器人实例
        :param component_name: 组件名称,用于指定机器人上的特定操控器
        :param wln_ratio: 权重比率,用于计算关节运动范围的阈值
        """
        self.rbt = robot
        self.component_name = component_name
        self.jlc_object = self.rbt.manipulator_dict[component_name].jlc
        self.wln_ratio = wln_ratio
        # 工作空间边界
        self.max_rng = 2.0  # 最大范围,单位为米

        # IK 宏定义
        wt_pos = 0.628  # 位置权重 0.628m->1 == 0.01->0.00628m
        wt_agl = 1 / (math.pi * math.pi)  # 角度权重  pi->1 == 0.01->0.18degree
        self.ws_wtlist = [wt_pos, wt_pos, wt_pos, wt_agl, wt_agl, wt_agl]

        # 最大可达范围
        self.jnt_bounds = np.array(self.rbt.get_jnt_ranges(component_name))

        # 提取关节的最小值和最大值以便快速访问
        self.jmvmin = self.jnt_bounds[:, 0]
        self.jmvmax = self.jnt_bounds[:, 1]

        self.jmvrng = self.jmvmax - self.jmvmin

        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio

    def set_jlc(self, jlc_name):
        """
        设置操控器组件名称并更新关节范围

        :param jlc_name: 操控器组件名称
        """
        self.component_name = jlc_name
        self.jnt_bounds = np.array(self.rbt.get_jnt_ranges(jlc_name))
        # 提取关节的最小值和最大值以便快速访问
        self.jmvmin = self.jnt_bounds[:, 0]
        self.jmvmax = self.jnt_bounds[:, 1]
        self.jmvrng = self.jmvmax - self.jmvmin
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio

    def jacobian(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算机器人操控器的雅可比矩阵

        :param tcp_jnt_id: 工具中心点的关节ID,仅接受单个值
        :param tcp_loc_pos: 工具中心点的局部位置
        :param tcp_loc_rotmat: 工具中心点的局部旋转矩阵
        :return: j, 一个 6xn 的 numpy 数组,表示雅可比矩阵

        author: weiwei
        date: 20161202, 20200331, 20200706
        """
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        j = np.zeros((6, len(self.jlc_object.tgtjnts)))
        counter = 0
        for jid in self.jlc_object.tgtjnts:
            grax = self.jlc_object.jnts[jid]["gl_motionax"]
            if self.jlc_object.jnts[jid]["type"] == 'revolute':
                diffq = tcp_gl_pos - self.jlc_object.jnts[jid]["gl_posq"]
                j[:3, counter] = np.cross(grax, diffq)
                j[3:6, counter] = grax
            if self.jlc_object.jnts[jid]["type"] == 'prismatic':
                j[:3, counter] = grax
            counter += 1
            if jid == tcp_jnt_id:
                break
        return j

    def _wln_weightmat(self, jntvalues):
        """
        获取 WLN 权重矩阵

        :param jntvalues: 关节值
        :return: 权重矩阵

        author: weiwei
        date: 20201126
        """
        wtmat = np.ones(self.jlc_object.ndof)
        # 最小阻尼区间
        selection = jntvalues < self.jmvmin_threshhold

        normalized_diff_at_selected = ((jntvalues - self.jmvmin) / (self.jmvmin_threshhold - self.jmvmin))[selection]
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)

        # 最大阻尼区间
        selection = jntvalues > self.jmvmax_threshhold

        normalized_diff_at_selected = ((self.jmvmax - jntvalues) / (self.jmvmax - self.jmvmax_threshhold))[selection]
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)

        wtmat[jntvalues >= self.jmvmax] = 0
        wtmat[jntvalues <= self.jmvmin] = 0

        return np.diag(wtmat)

    def manipulability(self, tcp_jnt_id):
        """
        计算 Yoshikawa 可操作性

        :param tcp_jnt_id: 工具中心点的关节ID,单个值或列表
        :return: 可操作性值

        author: weiwei
        date: 20200331
        """
        j = self.jacobian(tcp_jnt_id)
        return math.sqrt(np.linalg.det(np.dot(j, j.transpose())))

    def manipulability_axmat(self, tcp_jnt_id):
        """
        计算 Yasukawa 可操作性

        :param tcp_jnt_id: 工具中心点的关节ID,单个值或列表
        :return: 每列为可操作性的轴矩阵
        """
        armjac = self.jacobian(tcp_jnt_id)
        jjt = np.dot(armjac, armjac.T)
        pcv, pcaxmat = np.linalg.eig(jjt)
        # 仅保留平移部分
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:3, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:3, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:3, 2]
        return axmat

    def get_gl_tcp(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        获取全局工具中心点位姿
        :param tcp_jnt_id: 在 self.tgtjnts 中的关节ID
        :param tcp_loc_pos: 1x3 的 numpy 数组,描述在局部框架中的位置,单个值或列表
        :param tcp_loc_rotmat: 3x3 的 numpy 数组,描述在局部框架中的旋转矩阵,单个值或列表
        :return: 根据输入返回单个值或列表
        author: weiwei
        date: 20200706
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat
        tcp_gl_pos = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_pos) + \
                     self.jlc_object.jnts[tcp_jnt_id]["gl_posq"]
        tcp_gl_rotmat = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_rotmat)
        return tcp_gl_pos, tcp_gl_rotmat

    def tcp_error(self, tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算机器人末端与目标位置和旋转矩阵之间的误差

        注意:如果是列表,len(tgt_pos)=len(tgt_rotmat) <= len(tcp_jnt_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)

        :param tgt_pos: 目标位置向量(可以是单个值或关节ID的列表)
        :param tgt_rot: 目标旋转矩阵(可以是单个值或关节ID的列表)
        :param tcp_jnt_id: 在 self.tgtjnts 中的关节ID
        :param tcp_loc_pos: 1x3 的 numpy 数组,描述在局部框架中的位置,单个值或列表
        :param tcp_loc_rotmat: 3x3 的 numpy 数组,描述在局部框架中的旋转矩阵,单个值或列表
        :return: 一个 1x6 的 numpy 数组,前三个值表示位置的位移,后三个值表示旋转的位移

        author: weiwei
        date: 20180827, 20200331, 20200705
        """
        # 获取全局工具中心点位置和旋转矩阵
        tcp_globalpos, tcp_globalrotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        deltapw = np.zeros(6)
        # 计算位置误差
        deltapw[0:3] = (tgt_pos - tcp_globalpos)
        # 计算旋转误差
        deltapw[3:6] = rm.deltaw_between_rotmat(tgt_rot, tcp_globalrotmat.T)
        return deltapw

    def regulate_jnts(self):
        """
        检查给定的关节值是否在操作范围内

        超出范围的关节值将被拉回到它们的最大值
        :return: 两个参数,一个是布尔值,指示关节值是否在范围内
                 另一个是调整后的关节值.如果关节没有被调整,将返回相同的关节值

        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':
                # 如果关节的运动范围大于等于 360 度,则不需要调整
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] >= math.pi * 2:
                    rm.regulate_angle(self.jlc_object.jnts[id]['motion_rng'][0],
                                      self.jlc_object.jnts[id]['motion_rng'][1],
                                      self.jlc_object.jnts[id]["movement"])
            counter += 1

    def check_jntranges_drag(self, jntvalues):
        """
        检查给定的关节值是否在操作范围内, 超出范围的关节值将被拉回到它们的最大值

        :param jntvalues: 一个 1xn 的 numpy 数组
        :return: 两个参数,一个是布尔值,指示关节值是否在范围内
                 另一个是调整后的关节值.如果关节没有被调整,将返回相同的关节值

        author: weiwei
        date: 20161205
        """
        counter = 0
        isdragged = np.zeros_like(jntvalues)
        jntvaluesdragged = jntvalues.copy()
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':
                # 如果关节的运动范围小于 360 度,则需要调整
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] < math.pi * 2:
                    print("调整旋转关节 ！ Drag revolute")
                    if jntvalues[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jntvalues[counter] > \
                            self.jlc_object.jnts[id]['motion_rng'][1]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] +
                                                     self.jlc_object.jnts[id][
                                                         'motion_rng'][0]) / 2

            elif self.jlc_object.jnts[id]["type"] == 'prismatic':  # 平移关节
                print(" 调整平移关节 Drag prismatic")
                if jntvalues[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jntvalues[counter] > \
                        self.jlc_object.jnts[id]['motion_rng'][1]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] + self.jlc_object.jnts[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def num_ik(self,
               tgt_pos,
               tgt_rot,
               seed_jnt_values=None,
               tcp_jnt_id=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               local_minima="accept",
               toggle_debug=False):
        """
        使用 Levenberg-Marquardt 方法数值求解逆运动学

        注意:如果是列表,len(tgt_pos)=len(tgt_rotmat) <= len(tcp_jnt_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)

        :param tgt_pos: 目标位置,1x3的numpy数组
        :param tgt_rot: 目标旋转矩阵,3x3的numpy数组
        :param seed_jnt_values: 数值迭代的初始关节配置
        :param tcp_jnt_id: self.tgtjnts中的关节ID
        :param tcp_loc_pos: 在局部框架中描述的1x3 numpy数组,单个值或列表
        :param tcp_loc_rotmat: 在局部框架中描述的3x3 numpy数组,单个值或列表
        :param local_minima: 在局部极小值处的处理方式:"accept"、"randomrestart"或"end"
        :return: 1xn的numpy数组

        author: weiwei
        date: 20180203, 20200328
        """
        # 计算目标位置与当前末端位置的距离
        deltapos = tgt_pos - self.jlc_object.jnts[0]['gl_pos0']
        if np.linalg.norm(deltapos) > self.max_rng:
            wns.WarningMessage("目标超出最大范围！ The goal is outside maximum range!")
            return None

        # 设置默认的TCP关节ID和局部位置、旋转矩阵
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
            print(self.jlc_object.tcp_loc_pos)
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat

        # 备份当前关节值并初始化迭代关节值
        jntvalues_bk = self.jlc_object.get_jnt_values()
        jntvalues_iter = self.jlc_object.homeconf if seed_jnt_values is None else seed_jnt_values.copy()
        self.jlc_object.fk(jnt_values=jntvalues_iter)
        jntvalues_ref = jntvalues_iter.copy()

        # 初始化权重矩阵
        ws_wtdiagmat = np.diag(self.ws_wtlist)

        # 调试模式下的变量初始化
        if toggle_debug:
            if "jlm" not in dir():
                import robot_sim._kinematics.jlchain_mesh as jlm
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            # jlmgen = jlm.JntLnksMesh()

            # 初始化调试用的变量
            dqbefore = []  # 原始关节速度
            dqcorrected = []  # 修正后的关节速度
            dqnull = []  # 空间中的关节速度
            ajpath = []  # 关节路径

        errnormlast = 0.0
        errnormmax = 0.0

        # 迭代求解
        for i in range(1000):
            # 计算雅可比矩阵和误差
            j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            j1 = j[:3, :]
            j2 = j[3:6, :]
            err = self.tcp_error(tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            err_pos = err[:3]
            err_rot = err[3:6]
            errnorm_pos = err_pos.T.dot(err_pos)
            errnorm_rot = np.linalg.norm(err_rot)
            # if errnorm_rot < math.pi/6:
            #     err_rot = np.zeros(3)
            # errnorm_rot = 0
            errnorm = err.T.dot(ws_wtdiagmat).dot(err)
            # err = .05 / errnorm * err if errnorm > .05 else err

            # 更新最大误差
            if errnorm > errnormmax:
                errnormmax = errnorm
            # 调试信息输出
            if toggle_debug:
                print(errnorm_pos, errnorm_rot, errnorm)
                ajpath.append(self.jlc_object.get_jnt_values())

            # 检查误差是否在允许范围内
            if errnorm_pos < 1e-6 and errnorm_rot < math.pi / 6:
                if toggle_debug:
                    fig = plt.figure()
                    axbefore = fig.add_subplot(411)
                    axbefore.set_title('Original dq')
                    axnull = fig.add_subplot(412)
                    axnull.set_title('空域上的 dqref on Null space')
                    axcorrec = fig.add_subplot(413)
                    axcorrec.set_title('Minimized dq')
                    axaj = fig.add_subplot(414)
                    axbefore.plot(dqbefore)
                    axnull.plot(dqnull)
                    axcorrec.plot(dqcorrected)
                    axaj.plot(ajpath)
                    plt.show()
                # self.regulate_jnts()
                jntvalues_return = self.jlc_object.get_jnt_values()
                self.jlc_object.fk(jnt_values=jntvalues_bk)
                return jntvalues_return
            else:
                # 判断是否达到局部极小值
                if abs(errnorm - errnormlast) < 1e-12:
                    if toggle_debug:
                        fig = plt.figure()

                        # 添加第一个子图,用于显示原始的关节角度变化量
                        axbefore = fig.add_subplot(411)
                        axbefore.set_title('Original dq')  # 设置子图标题

                        # 添加第二个子图,用于显示空域上的关节角度变化量
                        axnull = fig.add_subplot(412)
                        axnull.set_title('dqref on Null space')  # 设置子图标题

                        # 添加第三个子图,用于显示最小化处理后的关节角度变化量
                        axcorrec = fig.add_subplot(413)
                        axcorrec.set_title('Minimized dq')  # 设置子图标题

                        # 添加第四个子图,用于显示调整后的关节路径
                        axaj = fig.add_subplot(414)

                        # 在第一个子图中绘制原始的关节角度变化量
                        axbefore.plot(dqbefore)

                        # 在第二个子图中绘制空域上的关节角度变化量
                        axnull.plot(dqnull)

                        # 在第三个子图中绘制最小化处理后的关节角度变化量
                        axcorrec.plot(dqcorrected)

                        # 在第四个子图中绘制调整后的关节路径
                        axaj.plot(ajpath)

                        plt.show()
                    if local_minima == 'accept':
                        wns.warn(
                            '绕过局部极小值！返回值是局部极小值,而不是精确的IK结果.')
                        jntvalues_return = self.jlc_object.get_jnt_values()
                        self.jlc_object.fk(jntvalues_bk)
                        return jntvalues_return
                    elif local_minima == 'randomrestart':
                        wns.warn('局部极小值！在局部极小值处随机重启.')
                        jntvalues_iter = self.jlc_object.rand_conf()
                        self.jlc_object.fk(jntvalues_iter)
                        continue
                    else:
                        print('没有可行的IK解!')
                        break
                else:
                    # -- notes --
                    ## note1: do not use np.linalg.inv since it is not precise
                    ## note2: use np.linalg.solve if the system is exactly determined, it is faster
                    ## note3: use np.linalg.lstsq if there might be singularity (no regularization)
                    ## see https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress
                    ## note4: null space https://www.slideserve.com/marietta/kinematic-redundancy
                    ## note5: avoid joint limits; Paper Name: Clamping weighted least-norm method for the manipulator kinematic control: Avoiding joint limits
                    ## note6: constant damper; Sugihara Paper: https://www.mi.ams.eng.osaka-u.ac.jp/member/sugihara/pub/jrsj_ik.pdf
                    # strecthingcoeff = 1 / (1 + math.exp(1 / ((errnorm / self.max_rng) * 1000 + 1)))
                    # strecthingcoeff = -2*math.pow(errnorm / errnormmax, 3)+3*math.pow(errnorm / errnormmax, 2)
                    # print("stretching ", strecthingcoeff)
                    # dampercoeff = (strecthingcoeff + .1) * 1e-6  # a non-zero regulation coefficient

                    # 计算阻尼系数
                    dampercoeff = 1e-3 * errnorm + 1e-6  # a非零调节系数
                    # -- lft moore-penrose inverse --
                    ## jtj = armjac.T.dot(armjac)
                    ## regulator = regcoeff*np.identity(jtj.shape[0])
                    ## jstar = np.linalg.inv(jtj+regulator).dot(armjac.T)
                    ## dq = jstar.dot(err)
                    # -- rgt moore-penrose inverse --
                    # # jjt
                    # jjt = j.dot(j.T)
                    # damper = dampercoeff * np.identity(jjt.shape[0])
                    # jsharp = j.T.dot(np.linalg.inv(jjt + damper))
                    # weighted jjt

                    # 计算加权雅可比矩阵的逆
                    qs_wtdiagmat = self._wln_weightmat(jntvalues_iter)

                    winv_j1t = np.linalg.inv(qs_wtdiagmat).dot(j1.T)
                    j1_winv_j1t = j1.dot(winv_j1t)
                    damper = dampercoeff * np.identity(j1_winv_j1t.shape[0])
                    j1sharp = winv_j1t.dot(np.linalg.inv(j1_winv_j1t + damper))
                    n1 = np.identity(jntvalues_ref.shape[0]) - j1sharp.dot(j1)
                    j2n1 = j2.dot(n1)
                    winv_j2n1t = np.linalg.inv(qs_wtdiagmat).dot(j2n1.T)
                    j2n1_winv_j2n1t = j2n1.dot(winv_j2n1t)
                    damper = dampercoeff * np.identity(j2n1_winv_j2n1t.shape[0])
                    j2n1sharp = winv_j2n1t.dot(np.linalg.inv(j2n1_winv_j2n1t + damper))

                    # 计算关节增量
                    err_pos = .1 * err_pos
                    # if errnorm_rot == 0:
                    dq = j1sharp.dot(err_pos)
                    dqref = (jntvalues_ref - jntvalues_iter)
                    dqref_on_ns = (np.identity(jntvalues_ref.shape[0]) - j1sharp.dot(j1)).dot(dqref)
                    dq_minimized = dq + dqref_on_ns

                    # else:
                    # err_rot = .1 * err_rot
                    # dq = j1sharp.dot(err_pos)+j2n1sharp.dot(err_rot-j2.dot(j1sharp.dot(err_pos)))
                    # dqref_on_ns = np.zeros(jntvalues_ref.shape[0])
                    # dq_minimized = dq

                    if toggle_debug:
                        dqbefore.append(dq)
                        dqcorrected.append(dq_minimized)
                        dqnull.append(dqref_on_ns)

                # 更新关节值
                jntvalues_iter += dq_minimized  # translation problem
                # isdragged, jntvalues_iter = self.check_jntsrange_drag(jntvalues_iter)
                # print(jntvalues_iter)
                self.jlc_object.fk(jnt_values=jntvalues_iter)
                # if toggle_debug:
                #     jlmgen.gensnp(jlinstance, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                #                   tcp_loc_rotmat=tcp_loc_rotmat, togglejntscs=True).reparentTo(base.render)

            errnormlast = errnorm

        # 如果迭代结束仍未找到解,输出调试信息
        if toggle_debug:
            fig = plt.figure()

            axbefore = fig.add_subplot(411)
            axbefore.set_title('Original dq')

            axnull = fig.add_subplot(412)
            axnull.set_title('dqref on Null space')

            axcorrec = fig.add_subplot(413)
            axcorrec.set_title('Minimized dq')

            axaj = fig.add_subplot(414)

            axbefore.plot(dqbefore)
            axnull.plot(dqnull)
            axcorrec.plot(dqcorrected)
            axaj.plot(ajpath)
            plt.show()
            self.jlc_object.gen_stickmodel(tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat, toggle_jntscs=True).attach_to(base)
            base.run()

        # 恢复原始关节值并返回None
        self.jlc_object.fk(jntvalues_bk)
        wns.warn('无法求解IK,返回None! Failed to solve the IK, returning None.')
        return None

    def numik_rel(self, deltapos, deltarotmat, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        在当前末端位置和姿态的基础上,添加增量位置和旋转矩阵,并计算新的逆运动学解

        :param deltapos: 增量位置,1x3的numpy数组
        :param deltarotmat: 增量旋转矩阵,3x3的numpy数组
        :param tcp_jnt_id: 关节ID,在self.tgtjnts中
        :param tcp_loc_pos: 在局部框架中描述的1x3 numpy数组,单个值或列表
        :param tcp_loc_rotmat: 在局部框架中描述的3x3 numpy数组,单个值或列表
        :return: 逆运动学解

        author: weiwei
        date: 20170412, 20200331
        """
        # 获取当前TCP的全局位置和旋转矩阵
        tcp_globalpos, tcp_globalrotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        # 计算目标位置和旋转矩阵
        tgt_pos = tcp_globalpos + deltapos
        tgt_rotmat = np.dot(deltarotmat, tcp_globalrotmat)
        # 获取当前关节配置
        start_conf = self.jlc_object.getjntvalues()
        # 调用numik方法计算新的逆运动学解
        return self.num_ik(tgt_pos, tgt_rotmat, start_conf=start_conf, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                           tcp_loc_rotmat=tcp_loc_rotmat)


if __name__ == '__main__':
    import time
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.robots.gofa5.gofa5_dh76 as gofa5
    import robot_sim.end_effectors.gripper.dh76.dh76 as dh

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    robot_s = gofa5.GOFA5(enable_cc=True)
    robot_s.gen_meshmodel().attach_to(base)
    # yumi_instance = ym.Yumi(enable_cc=True)
    component_name = 'arm'

    tgt_pos = np.array([.5, -.3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    niksolver = NIK(robot_s, component_name='arm')

    tic = time.time()
    jnt_values = niksolver.num_ik(tgt_pos, tgt_rotmat, toggle_debug=True)
    toc = time.time()
    print(toc - tic)

    robot_s.fk(component_name, jnt_values)
    yumi_meshmodel = robot_s.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)

    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print("result", result, toc - tic)
    base.run()
