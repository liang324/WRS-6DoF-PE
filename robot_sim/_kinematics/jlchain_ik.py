import math
import numpy as np
import basis.robot_math as rm


class JLChainIK(object):
    """
    逆向运动学类,用于计算关节空间中的运动学和雅可比矩阵.
    该类主要用于处理机械臂的逆向运动学(IK)问题,能够根据工具坐标、关节角度等信息计算出期望的姿态.
    """

    def __init__(self, jlc_object, wln_ratio=.05):
        """
        初始化逆向运动学计算类
        :param jlc_object: 一个 JLChain 类的对象,包含机器人结构的相关信息.
        :param wln_ratio: 关节运动范围的比例系数,控制运动范围的边界.
        """
        self.jlc_object = jlc_object  # 关联一个 JLChain 对象
        self.wln_ratio = wln_ratio  # 设置关节运动范围的比例系数

        # IK 宏定义
        wt_pos = 0.628  # 0.628米对应1 -> 0.01对应0.00628米
        wt_agl = 1 / (math.pi * math.pi)  # 角度比例,pi对应1 -> 0.01对应0.18度
        self.ws_wtlist = [wt_pos, wt_pos, wt_pos, wt_agl, wt_agl, wt_agl]  # 用于目标位置和角度的权重列表

        # 最大可达范围
        self.max_rng = 20.0

        # 提取关节的最小、最大范围以便快速访问
        self.jmvmin = self.jlc_object.jnt_ranges[:, 0]  # 最小关节范围
        self.jmvmax = self.jlc_object.jnt_ranges[:, 1]  # 最大关节范围
        self.jmvrng = self.jmvmax - self.jmvmin  # 关节运动范围
        self.jmvmiddle = (self.jmvmax + self.jmvmin) / 2  # 关节的中间值
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio  # 最小阈值
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio  # 最大阈值

    def _jacobian_sgl(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算单个关节的雅可比矩阵
        该函数计算指定关节的雅可比矩阵,返回一个 6xN 的矩阵
        其中 N 为关节数量,6 表示 3 个位置分量和 3 个旋转分量
        :param tcp_jnt_id: 工具中心姿态所在的关节 id,单个值
        :param tcp_loc_pos: 工具中心相对于工具坐标系的位置
        :param tcp_loc_rotmat: 工具中心相对于工具坐标系的旋转矩阵
        :return: 雅可比矩阵 j,一个 6xN 的 numpy 数组
        author: weiwei
        date: 20161202, 20200331, 20200706
        """
        # 通过工具中心的位置信息和旋转矩阵计算全局位置和旋转矩阵
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        # 初始化雅可比矩阵,大小为 6xN,其中 N 为目标关节数
        j = np.zeros((6, len(self.jlc_object.tgtjnts)))
        counter = 0
        # 遍历目标关节,计算雅可比矩阵
        for jid in self.jlc_object.tgtjnts:
            grax = self.jlc_object.jnts[jid]["gl_motionax"]  # 获取关节的运动轴
            # 如果是旋转关节,计算旋转轴和位置差的叉积
            if self.jlc_object.jnts[jid]["type"] == 'revolute':
                diffq = tcp_gl_pos - self.jlc_object.jnts[jid]["gl_posq"]  # 计算位置差
                j[:3, counter] = np.cross(grax, diffq)  # 叉积得到位置部分
                j[3:6, counter] = grax  # 旋转轴分量

            # 如果是平移关节,位置部分等于运动轴
            if self.jlc_object.jnts[jid]["type"] == 'prismatic':
                j[:3, counter] = grax  # 直接赋值为运动轴

            counter += 1

            # 如果当前关节是目标关节,则停止计算
            if jid == tcp_jnt_id:
                break
        return j

    def _wln_weightmat(self, jntvalues):
        """
        计算关节权重矩阵,用于计算关节的权重值,通常用于运动学优化
        :param jntvalues: 关节的当前值
        :return: 权重矩阵(对角矩阵),用于加权
        author: weiwei
        date: 20201126
        """
        # 初始化权重矩阵,大小与目标关节数相同,默认为 1
        wtmat = np.ones(len(self.jlc_object.tgtjnts))
        # 最小阻尼区间
        selection = jntvalues < self.jmvmin_threshhold  # 选择小于最小阈值的关节
        normalized_diff_at_selected = ((jntvalues - self.jmvmin) / (self.jmvmin_threshhold - self.jmvmin))[selection]
        # 更新权重矩阵,使用三次多项式进行平滑过渡
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
        # 最大阻尼区间
        selection = jntvalues > self.jmvmax_threshhold  # 选择大于最大阈值的关节
        normalized_diff_at_selected = ((self.jmvmax - jntvalues) / (self.jmvmax - self.jmvmax_threshhold))[selection]
        # 更新权重矩阵,使用三次多项式进行平滑过渡
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
        # 将超出最小或最大范围的关节的权重设为 0
        wtmat[jntvalues >= self.jmvmax] = 0
        wtmat[jntvalues <= self.jmvmin] = 0
        # 返回对角矩阵
        return np.diag(wtmat)

    def jacobian(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算机器人系统的雅可比矩阵.雅可比矩阵是描述关节速度和末端执行器速度之间关系的工具.
        支持多个 tcp_jnt_id(工具中心坐标),分别计算每个工具中心的雅可比矩阵并求和.
        :param tcp_jnt_id: 工具中心坐标的关节 id,可以是单个值或多个值的列表
        :param tcp_loc_pos: 工具相对于关节位置的偏移量,单个或多个位置
        :param tcp_loc_rotmat: 工具相对于关节的旋转矩阵,单个或多个旋转矩阵
        :return: 雅可比矩阵 j,大小为 (6 * len(tcp_jnt_id)) x n 的矩阵,其中 n 是目标关节数
        author: weiwei
        date: 20161202, 20200331, 20200706, 20201114
        """
        # 如果 tcp_jnt_id 是一个列表,则计算多个雅可比矩阵
        if isinstance(tcp_jnt_id, list):
            j = np.zeros((6 * (len(tcp_jnt_id)), len(self.jlc_object.tgtjnts)))
            # 对每个工具中心计算单个雅可比矩阵并将其放入大矩阵中
            for i, this_tcp_jnt_id in enumerate(tcp_jnt_id):
                j[6 * i:6 * i + 6, :] = self._jacobian_sgl(this_tcp_jnt_id, tcp_loc_pos[i], tcp_loc_rotmat[i])
            return j
        else:
            # 否则直接计算单个雅可比矩阵
            return self._jacobian_sgl(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)

    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        """
        计算 Yoshikawa 可操作性度量值,表示机器人末端执行器的可操作性或灵活性
        可操作性值越大,表示机器人能够进行更多的独立运动
        :param tcp_jnt_id: 工具中心坐标的关节 id,可以是单个值或多个值的列表
        :param tcp_loc_pos: 工具相对于关节位置的偏移量,单个或多个位置
        :param tcp_loc_rotmat: 工具相对于关节的旋转矩阵,单个或多个旋转矩阵
        :return: 可操作性度量值(标量),越大表示机器人可操作性越强
        author: weiwei
        date: 20200331
        """
        # 计算雅可比矩阵
        j = self.jacobian(tcp_jnt_id,
                          tcp_loc_pos,
                          tcp_loc_rotmat)
        # 计算可操作性度量值,使用雅可比矩阵的行列式的平方根
        return math.sqrt(np.linalg.det(np.dot(j, j.transpose())))

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat,
                             type="translational"):
        """
        计算机器人系统的雅可比矩阵的雅可比矩阵,计算雅可比矩阵的可操作性(Yoshikawa 可操作性)
        :param tcp_jnt_id: 工具中心姿态指定的关节ID,可以是单个值或列表
        :param type: 计算类型,"translational"表示平移,可用于描述平移可操作性,"rotational"表示旋转
        :return: 一个 3x3 的矩阵,每一列表示不同方向上的可操作性
        """
        # 计算雅可比矩阵
        j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        # 根据给定的类型(平移或旋转)选择雅可比矩阵的相关部分
        if type == "translational":
            jjt = np.dot(j[:3, :], j.transpose()[:, :3])  # 平移部分
        elif type == "rotational":
            jjt = np.dot(j[3:, :], j.transpose()[:, 3:])  # 旋转部分
        else:
            raise Exception("The parameter 'type' must be 'translational' or 'rotational'!")

        # 计算本征值和本征向量,作为可操作性的度量
        pcv, pcaxmat = np.linalg.eig(jjt)
        # 生成轴矩阵
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:, 2]
        return axmat

    def get_gl_tcp(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算给定的工具中心姿态的全局位置和全局旋转矩阵.
        :param tcp_jnt_id: 工具中心位置指定的关节ID
        :param tcp_loc_pos: 1x3 数组,描述在自定义关节坐标系下的位置,可以是单个值或列表
        :param tcp_loc_rotmat: 3x3 数组,描述在自定义关节坐标系下的旋转矩阵,可以是单个值或列表
        :return: 返回一个单一值或列表,取决于输入
        author: weiwei
        date: 20200706
        """
        # 如果没有提供工具中心关节ID,则使用默认值
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat
        # 如果 tcp_jnt_id 是一个列表,计算多个工具中心的位置和旋转矩阵
        if isinstance(tcp_jnt_id, list):
            returnposlist = []
            returnrotmatlist = []
            for i, jid in enumerate(tcp_jnt_id):
                tcp_gl_pos = np.dot(self.jlc_object.jnts[jid]["gl_rotmatq"], tcp_loc_pos[i]) + \
                             self.jlc_object.jnts[jid]["gl_posq"]
                tcp_gl_rotmat = np.dot(self.jlc_object.jnts[jid]["gl_rotmatq"], tcp_loc_rotmat[i])
                returnposlist.append(tcp_gl_pos)
                returnrotmatlist.append(tcp_gl_rotmat)
            return [returnposlist, returnrotmatlist]
        else:
            # 否则计算单个工具中心的位置和旋转矩阵
            tcp_gl_pos = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_pos) + \
                         self.jlc_object.jnts[tcp_jnt_id]["gl_posq"]
            tcp_gl_rotmat = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_rotmat)
            return tcp_gl_pos, tcp_gl_rotmat

    def tcp_error(self, tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        计算机器人末端执行器 当前位置 与 目标位置 和 目标旋转矩阵 之间的误差
        :param tgt_pos: 目标位置向量,可以是单个值或位置列表
        :param tgt_rot: 目标旋转矩阵,可以是单个值或旋转矩阵列表
        :param tcp_jnt_id: 工具中心位置指定的关节ID
        :param tcp_loc_pos: 1x3 数组,描述在自定义关节坐标系下的位置,可以是单个值或列表
        :param tcp_loc_rotmat: 3x3 数组,描述在自定义关节坐标系下的旋转矩阵,可以是单个值或列表
        :return: 返回一个 1x6 的数组,前三个值表示位置上的位移,后三个值表示旋转上的位移
        author: weiwei
        date: 20180827, 20200331, 20200705
        """
        # 获取工具中心的全局位置和旋转矩阵
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        # 如果目标位置是列表,则为每个目标位置计算误差
        if isinstance(tgt_pos, list):
            deltapw = np.zeros(6 * len(tgt_pos))
            for i, this_tgt_pos in enumerate(tgt_pos):
                deltapw[6 * i:6 * i + 3] = (this_tgt_pos - tcp_gl_pos[i])
                deltapw[6 * i + 3:6 * i + 6] = rm.deltaw_between_rotmat(tcp_gl_rotmat[i], tgt_rot[i])
            return deltapw
        else:
            # 如果目标位置是单个值,则直接计算误差
            deltapw = np.zeros(6)
            deltapw[0:3] = (tgt_pos - tcp_gl_pos)
            deltapw[3:6] = rm.deltaw_between_rotmat(tcp_gl_rotmat, tgt_rot)
            return deltapw

    def regulate_jnts(self):
        """
        检查给定的关节值是否在操作范围内.如果关节值超出范围,会将其拉回到最大值.

        :return: 返回两个参数:
                1. 一个布尔值,表示关节值是否在范围内
                2. 一个关节值数组,表示拖动后的关节值.如果没有拖动,返回相同的关节值
        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':  # 如果是旋转关节
                # 如果关节的运动范围大于 2π,表示可以完全旋转
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] >= math.pi * 2:
                    rm.regulate_angle(self.jlc_object.jnts[id]['motion_rng'][0],
                                      self.jlc_object.jnts[id]['motion_rng'][1],
                                      self.jlc_object.jnts[id]["movement"])
            counter += 1

    def check_jntranges_drag(self, jnt_values):
        """
        检查给定的关节值是否在操作范围内.如果关节值超出范围,会将其拉回到最大值

        :param jnt_values: 一个 1xn 的 NumPy ndarray,表示关节值
        :return: 返回两个参数:
                1. 一个布尔值数组,表示每个关节值是否被拖动(超出范围)
                2. 一个关节值数组,表示拖动后的关节值.如果没有拖动,返回相同的关节值
        author: weiwei
        date: 20161205
        """
        counter = 0
        isdragged = np.zeros_like(jnt_values)  # 创建一个与 jnt_values 同样大小的零数组
        jntvaluesdragged = jnt_values.copy()  # 复制原始关节值
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':  # 如果是旋转关节
                # 如果关节的运动范围小于 2π
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] < math.pi * 2:
                    # if jntvalues[counter] < jlinstance.jnts[id]['motion_rng'][0]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][0]
                    # elif jntvalues[counter] > jlinstance.jnts[id]['motion_rng'][1]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][1]
                    # 检查关节值是否超出范围,若超出,则拖动到范围的中间值
                    print("Drag revolute")
                    if jnt_values[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jnt_values[counter] > \
                            self.jlc_object.jnts[id]['motion_rng'][1]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] +
                                                     self.jlc_object.jnts[id][
                                                         'motion_rng'][0]) / 2

            elif self.jlc_object.jnts[id]["type"] == 'prismatic':  # 如果是直线关节
                # if jntvalues[counter] < jlinstance.jnts[id]['motion_rng'][0]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][0]
                # elif jntvalues[counter] > jlinstance.jnts[id]['motion_rng'][1]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][1]

                # 检查关节值是否超出范围,若超出,则拖动到范围的中间值
                print("Drag prismatic")
                if jnt_values[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jnt_values[counter] > \
                        self.jlc_object.jnts[id]['motion_rng'][1]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] + self.jlc_object.jnts[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def num_ik(self,
               tgt_pos,
               tgt_rot,
               seed_jnt_values=None,
               max_niter=100,
               tcp_jnt_id=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               local_minima="randomrestart",
               toggle_debug=False):
        """
        使用 Levenberg-Marquardt 方法数值求解逆向运动学(IK)

        :param tgt_pos: 目标位置,1x3 的 numpy ndarray
        :param tgt_rot: 目标方向,3x3 的 numpy ndarray
        :param seed_jnt_values: 数值迭代的初始关节配置
        :param max_niter: 最大迭代次数
        :param tcp_jnt_id: 目标关节 ID(如果为空,则使用默认值)
        :param tcp_loc_pos: TCP 相对于目标关节的局部位置,1x3 numpy ndarray,可以是单值或列表
        :param tcp_loc_rotmat: TCP 相对于目标关节的局部旋转矩阵,3x3 numpy ndarray,可以是单值或列表
        :param local_minima: 到达局部最小值时的处理方式:"accept", "randomrestart", "end"
        :param toggle_debug: 是否启用调试模式
        :return: 返回求解的关节值(1xN numpy ndarray),如果无法求解,则返回 None
        author: weiwei
        date: 20180203, 20200328
        """
        # 计算目标位置与当前末端执行器位置的差值
        deltapos = tgt_pos - self.jlc_object.jnts[0]['gl_pos0']

        # 如果目标超出最大工作范围,返回 None
        if np.linalg.norm(deltapos) > self.max_rng:
            print("The goal is outside maximum range!")
            return None

        # 若未提供 tcp_jnt_id,使用默认值
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id

        # 若未提供 tcp_loc_pos 或 tcp_loc_rotmat,使用默认值
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat

        # 如果目标位置是列表,调整相关参数的长度
        if isinstance(tgt_pos, list):
            tcp_jnt_id = tcp_jnt_id[0:len(tgt_pos)]
            tcp_loc_pos = tcp_loc_pos[0:len(tgt_pos)]
            tcp_loc_rotmat = tcp_loc_rotmat[0:len(tgt_pos)]
        elif isinstance(tcp_jnt_id, list):
            tcp_jnt_id = tcp_jnt_id[0]
            tcp_loc_pos = tcp_loc_pos[0]
            tcp_loc_rotmat = tcp_loc_rotmat[0]

        # 获取当前关节值
        jnt_values_bk = self.jlc_object.get_jnt_values()

        # 设置初始关节值
        jnt_values_iter = self.jlc_object.homeconf if seed_jnt_values is None else seed_jnt_values.copy()

        # 执行正向运动学计算
        self.jlc_object.fk(jnt_values=jnt_values_iter)
        jnt_values_ref = jnt_values_iter.copy()

        # 构建权重矩阵
        if isinstance(tcp_jnt_id, list):
            diaglist = []
            for i in tcp_jnt_id:
                diaglist += self.ws_wtlist
            ws_wtdiagmat = np.diag(diaglist)
        else:
            ws_wtdiagmat = np.diag(self.ws_wtlist)

        # 启动调试模式时的额外设置
        if toggle_debug:
            if "jlm" not in dir():
                import robot_sim._kinematics.jlchain_mesh as jlm
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []
        random_restart = False
        errnormlast = 0.0
        errnormmax = 0.0

        # 开始迭代
        for i in range(max_niter):
            # 计算雅可比矩阵
            j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            # 计算末端执行器的误差
            err = self.tcp_error(tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            errnorm = err.T.dot(ws_wtdiagmat).dot(err)
            # err = .05 / errnorm * err if errnorm > .05 else err
            if errnorm > errnormmax:
                errnormmax = errnorm
            # 输出调试信息
            if toggle_debug:
                print(errnorm)
                ajpath.append(self.jlc_object.get_jnt_values())
            # 若误差足够小,则返回关节值
            if errnorm < 1e-9:
                if toggle_debug:
                    print(f"找到结果前的 IK 迭代次数 Number of IK iterations before finding a result: {i}")
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
                jntvalues_return = self.jlc_object.get_jnt_values()
                self.jlc_object.fk(jnt_values=jnt_values_bk)
                return jntvalues_return
            else:
                # 判断是否陷入局部最小值
                if abs(errnorm - errnormlast) < 1e-12:
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

                    # 局部最小值处理
                    if local_minima == 'accept':
                        print('绕过局部最小值！返回值是局部最小值,而不是精确的 IK 结果 Bypassing local minima! The return value is a local minima, not an exact IK result.')
                        jntvalues_return = self.jlc_object.get_jnt_values()
                        self.jlc_object.fk(jnt_values_bk)
                        return jntvalues_return
                    elif local_minima == 'randomrestart':
                        print('局部最小值！在局部最小值随机重启 Local Minima! Random restart at local minima!')
                        jnt_values_iter = self.jlc_object.rand_conf()
                        self.jlc_object.fk(jnt_values_iter)
                        random_restart = True
                        continue
                    else:
                        print('没有可行的 IK 解决方案 No feasible IK solution!')
                        break
                else:
                    dampercoeff = 1e-3 * errnorm + 1e-6  # 非零调节系数
                    qs_wtdiagmat = self._wln_weightmat(jnt_values_iter)

                    w_jt = qs_wtdiagmat.dot(j.T)
                    j_w_jt = j.dot(w_jt)
                    damper = dampercoeff * np.identity(j_w_jt.shape[0])
                    jsharp = w_jt.dot(np.linalg.inv(j_w_jt + damper))
                    # Clamping (Paper Name: Clamping weighted least-norm method for the manipulator kinematic control)
                    phi_q = ((2 * jnt_values_iter - self.jmvmiddle) / self.jmvrng)
                    clamping = -(np.identity(qs_wtdiagmat.shape[0]) - qs_wtdiagmat).dot(phi_q)
                    # # if do not use WLN
                    # j_jt = j.dot(j.T)
                    # damper = dampercoeff * np.identity(j_jt.shape[0])
                    # jsharp = j.T.dot(np.linalg.inv(j_jt + damper))
                    # update dq
                    # 计算加权最小二乘法结果
                    dq = .1 * jsharp.dot(err)
                    if not random_restart:
                        w_init = 0.1
                    else:
                        w_init = 0
                    w_middle = 1
                    ns_projmat = np.identity(jnt_values_iter.size) - jsharp.dot(j)
                    dqref_init = (jnt_values_ref - jnt_values_iter)
                    dqref_on_ns = ns_projmat.dot(w_init * dqref_init + w_middle * clamping)
                    dq_minimized = dq + dqref_on_ns
                    # 调试时记录信息
                    if toggle_debug:
                        dqbefore.append(dq)
                        dqcorrected.append(dq_minimized)
                        dqnull.append(dqref_on_ns)
                jnt_values_iter += dq_minimized  # 更新关节值
                # isdragged, jntvalues_iter = self.check_jntsrange_drag(jntvalues_iter)
                # print(jnt_values_iter)
                self.jlc_object.fk(jnt_values=jnt_values_iter)
                # if toggle_debug:
                #     self.jlc_object.gen_stickmodel(tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                #                                    tcp_loc_rotmat=tcp_loc_rotmat, toggle_jntscs=True).attach_to(base)
            errnormlast = errnorm
        # 调试信息显示
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
            # base.run()
        # 恢复初始关节值并返回 None
        self.jlc_object.fk(jnt_values_bk)
        print('无法解决 IK,返回 None ！ Failed to solve the IK, returning None.')
        return None

    def numik_rel(self, deltapos, deltarotmat, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        将增量位置(deltapos)和增量旋转矩阵(deltarotmat)添加到当前末端执行器的位姿,然后调用 `numik` 方法计算逆向运动学解

        :param deltapos: 增量位置(1x3 numpy ndarray)
        :param deltarotmat: 增量旋转矩阵(3x3 numpy ndarray)
        :param tcp_jnt_id: 目标关节的 ID,位于 `self.tgtjnts`
        :param tcp_loc_pos: 相对于目标关节的局部位置,1x3 numpy ndarray,可能是单个值或列表
        :param tcp_loc_rotmat: 相对于目标关节的局部旋转矩阵,3x3 numpy ndarray,可能是单个值或列表
        :return: 返回 `numik` 方法的结果,即计算得到的逆向运动学解
        author: weiwei
        date: 20170412, 20200331
        """
        # 获取当前末端执行器的全局位置和旋转矩阵
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)

        if isinstance(tcp_jnt_id, list):  # 处理多个目标关节的情况
            tgt_pos = []
            tgt_rotmat = []

            # 遍历每个目标关节,计算新的目标位置和目标旋转矩阵
            for i, jid in enumerate(tcp_jnt_id):
                tgt_pos.append(tcp_gl_pos[i] + deltapos[i])  # 将增量位置添加到当前全局位置
                tgt_rotmat.append(np.dot(deltarotmat, tcp_gl_rotmat[i]))  # 将增量旋转矩阵与当前旋转矩阵相乘
            # 获取初始关节值
            start_conf = self.jlc_object.getjntvalues()
            # return numik(rjlinstance, tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat)
        else:
            # 只有单一目标关节
            tgt_pos = tcp_gl_pos + deltapos  # 计算新的目标位置
            tgt_rotmat = np.dot(deltarotmat, tcp_gl_rotmat)  # 计算新的目标旋转矩阵
            start_conf = self.jlc_object.getjntvalues()  # 获取初始关节值
        # 调用 numik 方法计算逆向运动学解
        return self.num_ik(tgt_pos, tgt_rotmat, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                           tcp_loc_rotmat=tcp_loc_rotmat)
