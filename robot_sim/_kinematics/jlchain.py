import math
import copy
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain_mesh as jlm
import robot_sim._kinematics.jlchain_ik as jlik

try:
    is_jlik_fast = False
    import robot_sim._kinematics.jlik_fast as jlik_fast
except:
    is_jlik_fast = False


# try:
#     import cupy as cp
#     is_gpu_acc = True
#     np.dot = lambda a,b: cp.asnumpy(cp.asarray(a)@ cp.asarray(b))
#     # np.cross = lambda a, b: (cp.asarray(a) @ cp.asarray(b)).asnumpy()
#     print("CPU acceleration OPEN")
# except:
#     import warnings
#     warnings.warn("No GPU acceleration for product")
#     is_gpu_acc = False


class JLChain(object):
    """
    关节-连杆链(Joint Link Chain),不允许有分支
    
    使用方法:
    1. 继承此类并重载 self._initjntlnks()/self.tgtjnts 来定义新的关节-连杆链
    2. 定义多个此类的实例以组成复杂的结构
    
    注意:
    关节类型包括 "revolute"(旋转关节)、"prismatic"(平移关节)和 "end"(末端关节)；一个 `JLChain` 对象总是包含两个 "end" 关节
    """
    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 homeconf=np.zeros(6),
                 name='jlchain',
                 cdprimitive_type='box',
                 cdmesh_type='triangles'):
        """
        初始化一个机械臂
        
        命名规则:
        allvalues -- 所有值:包括基座和末端的固定关节的所有关节值(两个关节值为0)
        conf -- 配置:目标关节值
        :param pos: 机械臂位置
        :param rotmat: 机械臂旋转矩阵
        :param homeconf: 默认关节配置
        :param name: 机械臂名称
        :param cdprimitive_type: 碰撞模型的类型('aabb', 'obb', 'convex_hull', 'triangulation'等)
        :param cdmesh_type: 碰撞网格类型
        :return:
        """
        self.name = name  # 机械臂名称
        self.pos = pos  # 初始位置
        self.rotmat = rotmat  # 初始旋转矩阵
        self.ndof = homeconf.shape[0]  # 关节自由度数
        self._zeroconf = np.zeros(self.ndof)  # 默认零配置
        self._homeconf = homeconf.astype('float64')  # 转换为 float64 类型

        # 初始化关节和连杆
        self.lnks, self.jnts = self._init_jlchain()  # 初始化连杆和关节链
        self._tgtjnts = list(range(1, self.ndof + 1))  # 目标关节列表
        self._jnt_ranges = self._get_jnt_ranges()  # 获取关节范围
        self.goto_homeconf()  # 转到初始配置

        # 默认TCP(工具坐标系)
        self.tcp_jnt_id = -1  # 默认TCP关节ID为-1
        self.tcp_loc_pos = np.zeros(3)  # 默认位置为零
        self.tcp_loc_rotmat = np.eye(3)  # 默认旋转矩阵为单位矩阵

        # 碰撞模型
        self.cdprimitive_type = cdprimitive_type  # 碰撞原语类型
        self.cdmesh_type = cdmesh_type  # 碰撞网格类型
        self._mt = jlm.JLChainMesh(self, cdprimitive_type=cdprimitive_type, cdmesh_type=cdmesh_type)  # 机械臂网格
        self._ikt = jlik.JLChainIK(self)  # 机械臂的逆向运动学

    def _init_jlchain(self):
        """
        初始化关节和连杆链
        
        该方法返回两个列表,第一个是关节列表,第二个是连杆列表
        连杆:包含连杆属性的字典列表
        关节:包含关节属性的字典列表
        假设关节数量等于连杆数量加1
        关节 i 连接连杆 i-1 和 i

        :return: 返回连杆和关节列表
        author: weiwei
        date: 20161202tsukuba, 20190328toyonaka, 20200330toyonaka
        """
        lnks = [dict() for i in range(self.ndof + 1)]  # 初始化连杆列表
        jnts = [dict() for i in range(self.ndof + 2)]  # 初始化关节列表

        # 初始化每个连杆的属性
        for id in range(self.ndof + 1):
            lnks[id]['name'] = 'link0'
            lnks[id]['loc_pos'] = np.array([0, 0, 0])  # 初始位置
            lnks[id]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)  # 初始旋转矩阵
            lnks[id]['com'] = np.zeros(3)  # 连杆的质心
            lnks[id]['inertia'] = np.eye(3)  # 连杆的惯性矩阵
            lnks[id]['mass'] = 0  # 连杆的质量
            lnks[id]['mesh_file'] = None  # 网格文件(没有网格时为 None)
            lnks[id]['collision_model'] = None  # 碰撞模型
            lnks[id]['cdprimit_childid'] = -1  # 碰撞检测子ID
            lnks[id]['scale'] = [1, 1, 1]  # 连杆的尺度 3 list
            lnks[id]['rgba'] = [.7, .7, .7, 1]  # 连杆的颜色(RGBA)4 list

        # 初始化每个关节的属性
        for id in range(self.ndof + 2):
            jnts[id]['type'] = 'revolute'  # 默认关节类型为旋转关节
            jnts[id]['parent'] = id - 1  # 父关节
            jnts[id]['child'] = id + 1  # 子关节
            jnts[id]['loc_pos'] = np.array([0, .1, 0]) if id > 0 else np.array([0, 0, 0])  # 关节位置
            jnts[id]['loc_rotmat'] = np.eye(3)  # 关节旋转矩阵
            jnts[id]['loc_motionax'] = np.array([0, 0, 1])  # 旋转关节的旋转轴,平移关节的线性轴
            jnts[id]['gl_pos0'] = jnts[id]['loc_pos']  # 更新的位置 by self._update_fk
            jnts[id]['gl_rotmat0'] = jnts[id]['loc_rotmat']  # 新的旋转矩阵 by self._update_fk
            jnts[id]['gl_motionax'] = jnts[id]['loc_motionax']  # # 更新的运动轴 by self._update_fk
            jnts[id]['gl_posq'] = jnts[id]['gl_pos0']  # 更新的全局位置 by self._update_fk
            jnts[id]['gl_rotmatq'] = jnts[id]['gl_rotmat0']  # 更新的全局旋转矩阵 by self._update_fk
            jnts[id]['motion_rng'] = [-math.pi, math.pi]  # 关节运动范围(最小值和最大值)
            jnts[id]['motion_val'] = 0  # 关节的当前值

        # 设置第一个关节为末端关节(end)
        jnts[0]['gl_pos0'] = self.pos  # 基本位置
        jnts[0]['gl_rotmat0'] = self.rotmat  # 基本旋转矩阵
        jnts[0]['type'] = 'end'  # 设置为末端关节
        jnts[self.ndof + 1]['loc_pos'] = np.array([0, 0, 0])  # 设置末端关节的位置
        jnts[self.ndof + 1]['child'] = -1  # 末端关节没有子关节
        jnts[self.ndof + 1]['type'] = 'end'  # 末端关节没有子关节
        # 返回初始化后的连杆和关节列表
        return lnks, jnts

    def _update_fk(self):
        """
        更新运动学
        
        注意:这个函数不应该被显式调用
        
        它会通过如 movexxx 之类的函数自动调用
        :return: 更新后的连杆和关节
        author: weiwei
        date: 20161202, 20201009osaka
        """
        if is_jlik_fast:
            # 如果启用了快速运动学(jlik_fast),则调用该方法
            jlik_fast._update_fk(self)
        else:
            id = 0
            while id != -1:
                # 更新关节值
                pjid = self.jnts[id]['parent']  # 获取当前关节的父关节ID
                if pjid == -1:
                    # 如果父关节ID为-1,说明这是根关节
                    self.jnts[id]['gl_pos0'] = self.pos  # 设置全局位置
                    self.jnts[id]['gl_rotmat0'] = self.rotmat  # 设置全局旋转矩阵
                else:
                    # 否则,使用父关节的位置和旋转矩阵计算当前关节的位置和旋转矩阵
                    self.jnts[id]['gl_pos0'] = self.jnts[pjid]['gl_posq'] + np.dot(self.jnts[pjid]['gl_rotmatq'],
                                                                                   self.jnts[id]['loc_pos'])
                    self.jnts[id]['gl_rotmat0'] = np.dot(self.jnts[pjid]['gl_rotmatq'], self.jnts[id]['loc_rotmat'])

                # 计算关节的旋转轴
                self.jnts[id]['gl_motionax'] = np.dot(self.jnts[id]['gl_rotmat0'], self.jnts[id]['loc_motionax'])

                # 根据关节类型(末端、固定、旋转、平移)更新关节的全局位置和旋转矩阵
                if self.jnts[id]['type'] == "end" or self.jnts[id]['type'] == "fixed":
                    self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                    self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
                elif self.jnts[id]['type'] == "revolute":
                    # 如果是旋转关节,则根据关节的旋转角度更新旋转矩阵
                    self.jnts[id]['gl_rotmatq'] = np.dot(self.jnts[id]['gl_rotmat0'],
                                                         rm.rotmat_from_axangle(self.jnts[id]['loc_motionax'],
                                                                                self.jnts[id]['motion_val']))
                    self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
                elif self.jnts[id]['type'] == "prismatic":
                    # 如果是平移关节,则根据关节的平移量更新位置
                    self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                    tmp_translation = np.dot(self.jnts[id]['gl_rotmatq'],
                                             self.jnts[id]['loc_motionax'] * self.jnts[id]['motion_val'])
                    self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0'] + tmp_translation

                # 更新连杆的全局位置和旋转矩阵
                if id < self.ndof + 1:
                    self.lnks[id]['gl_pos'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_pos']) + \
                                              self.jnts[id]['gl_posq']
                    self.lnks[id]['gl_rotmat'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_rotmat'])
                    # self.lnks[id]['cdprimit_cache'][0] = True

                id = self.jnts[id]['child']  # 移动到下一个关节(即当前关节的子关节)
        return self.lnks, self.jnts  # 返回更新后的连杆和关节列表

    @property
    def homeconf(self):
        """
        获取机械臂的初始关节配置(零位)

        :return: 返回一个包含每个关节初始位置的数组
        """
        return np.array([self._homeconf[i - 1] for i in self.tgtjnts])

    @property
    def zeroconf(self):
        """
        获取机械臂的零位关节配置
        
        :return: 返回一个包含零位关节配置的数组
        """
        return np.array([self._zeroconf[i - 1] for i in self.tgtjnts])

    @property
    def tgtjnts(self):
        """
        获取目标关节列表
        
        :return: 返回目标关节列表
        """
        return self._tgtjnts

    @property
    def jnt_ranges(self):
        """
        获取关节的运动范围
        
        :return: 返回包含每个关节最小值和最大值的数组
        """
        return self._jnt_ranges

    @tgtjnts.setter
    def tgtjnts(self, values):
        """
        设置目标关节列表,并更新关节的运动范围
        
        :param values: 新的目标关节列表
        """
        self._tgtjnts = values
        self._jnt_ranges = self._get_jnt_ranges()  # 更新关节范围
        self._ikt = jlik.JLChainIK(self)  # 重新初始化反向运动学解算器

    def _get_jnt_ranges(self):
        """
        获取关节的运动范围
        
        :return: 返回关节运动范围的二维数组,每个关节有最小值和最大值
        date: 20180602, 20200704osaka
        author: weiwei
        """
        if self.tgtjnts:
            jnt_limits = []
            for id in self.tgtjnts:
                # 获取每个关节的运动范围
                jnt_limits.append(self.jnts[id]['motion_rng'])
            return np.asarray(jnt_limits)  # 返回关节范围数组
        else:
            return np.empty((0, 2))  # 如果没有目标关节,返回空数组

    def fix_to(self, pos, rotmat, jnt_values=None):
        """
        将链条末端固定到给定的位置和旋转矩阵上
        
        :param pos: 目标位置
        :param rotmat: 目标旋转矩阵
        :param jnt_values: 可选的关节值
        :return: 返回更新后的连杆和关节
        """
        self.pos = pos
        self.rotmat = rotmat
        return self.fk(jnt_values=jnt_values)

    def set_homeconf(self, jnt_values=None):
        """
        设置机械臂的初始关节配置(零位)
        
        :param jnt_values: 可选的初始关节配置值,默认是零
        :return:
        """
        if jnt_values is None:
            jnt_values = np.zeros(self.ndof)  # 如果没有传入值,默认使用零配置
        if len(jnt_values) == self.ndof:
            self._homeconf = jnt_values  # 设置初始配置
        else:
            print('The given values must have enough dof!')
            raise Exception

    def reinitialize(self, cdprimitive_type=None, cdmesh_type=None):
        """
        重新初始化关节链条,通过更新运动学和重建关节网格
        
        :param cdprimitive_type: 可选的碰撞体类型
        :param cdmesh_type: 可选的碰撞网格类型
        :return:
        author: weiwei
        date: 20201126
        """
        self._jnt_ranges = self._get_jnt_ranges()  # 更新关节范围
        self.goto_homeconf()  # 将关节恢复到初始配置
        if cdprimitive_type is None:  # 如果没有传入碰撞体类型,使用默认值
            cdprimitive_type = self.cdprimitive_type
        if cdmesh_type is None:  # 如果没有传入碰撞网格类型,使用默认值
            cdmesh_type = self.cdmesh_type
        self._mg = jlm.JLChainMesh(self, cdprimitive_type, cdmesh_type)  # 创建新的网格
        self._ikt = jlik.JLChainIK(self)  # 重新初始化逆运动学解算器

    def set_tcp(self, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        设置 TCP(末端执行器)的位置、旋转矩阵和关节ID
        
        :param tcp_jnt_id: 可选的TCP关节ID
        :param tcp_loc_pos: 可选的TCP位置
        :param tcp_loc_rotmat: 可选的TCP旋转矩阵
        """
        if tcp_jnt_id is not None:
            self.tcp_jnt_id = tcp_jnt_id
        if tcp_loc_pos is not None:
            self.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self,
                   tcp_jnt_id=None,
                   tcp_loc_pos=None,
                   tcp_loc_rotmat=None):
        """
        获取末端执行器(TCP)的工具中心姿态(TCP pose)
        
        tcp_jnt_id、tcp_loc_pos 和 tcp_loc_rotmat 是工具中心姿态的参数,
        它们用于临时计算.如果没有提供这些参数,self.tcp_xxx 参数将不会改变,
        会使用默认的 self.tcp_jnt_id、self.tcp_loc_pos 和 self.tcp_loc_rotmat
        :param tcp_jnt_id: 末端执行器的关节ID(可选)
        :param tcp_loc_pos: 末端执行器的位置(可选)
        :param tcp_loc_rotmat: 末端执行器的旋转矩阵(可选)
        :return: 返回计算得到的末端执行器姿态(TCP)
        """
        return self._ikt.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)

    def is_jnt_values_in_ranges(self, jnt_values):
        """
        检查给定的关节值是否在允许的范围内
        
        :param jnt_values: 关节值列表或数组
        :return: 如果所有关节值都在有效范围内,返回 True,否则返回 False
        author: weiwei
        date: 20220326toyonaka
        """
        jnt_values = np.asarray(jnt_values)
        if np.all(self.jnt_ranges[:, 0] <= jnt_values) and np.all(jnt_values <= self.jnt_ranges[:, 1]):
            return True
        else:
            return False

    def fk(self, jnt_values=None):
        """
        使用正向运动学(Forward Kinematics,FK)来移动关节
        
        :param jnt_values: 一个 1xN 的 ndarray,每个元素表示一个关节的值(弧度或米)
        :return: 返回移动是否成功,若超出运动范围,返回 "out_of_rng",否则返回 "succ"
        author: weiwei
        date: 20161205, 20201009osaka
        """
        status = "succ" # "succ" 或 "out_of_rng"
        if jnt_values is not None:
            counter = 0
            for id in self.tgtjnts:
                if jnt_values[counter] < self.jnts[id]["motion_rng"][0] or jnt_values[counter] > \
                        self.jnts[id]["motion_rng"][1]:
                    status = "out_of_rng"
                self.jnts[id]['motion_val'] = jnt_values[counter]
                counter += 1
        self._update_fk()
        return status

    def goto_homeconf(self):
        """
        将机器人移动到初始姿态(零位配置)
        
        :return: 无返回值
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.homeconf)

    def goto_zeroconf(self):
        """
        将机器人移动到零位姿态(默认配置)
        
        :return: 无返回值
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.zeroconf)

    def get_jnt_values(self):
        """
        获取当前的关节值
        
        :return: 返回当前关节值的 1xN ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for id in self.tgtjnts:
            jnt_values[counter] = self.jnts[id]['motion_val']
            counter += 1
        return jnt_values

    def rand_conf(self):
        """
        生成一个随机的配置
        
        :return: 返回一个包含随机关节值的 1xN ndarray
        author: weiwei
        date: 20200326
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for i in self.tgtjnts:
            jnt_values[counter] = np.random.uniform(self.jnts[i]['motion_rng'][0], self.jnts[i]['motion_rng'][1])
            counter += 1
        return jnt_values

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           max_niter=100,
           local_minima="accept",
           toggle_debug=False):
        """
        逆向运动学求解

        注意1:在 rotjntlinksik 的 numik 函数中,tcp_jnt_id、tcp_loc_pos、tcp_loc_rotmat 是工具中心姿态(TCP pose)的参数.
        它们用于临时计算,如果未提供这些参数,将使用 self.tcp_xxx 参数.
        注意2:如果列表的长度不一致,len(tgt_pos) = len(tgt_rotmat) < len(tcp_jnt_id) = len(tcp_loc_pos) = len(tcp_loc_rotmat)

        :param tgt_pos: 目标位置,1x3 的 nparray,可以是单一值或列表
        :param tgt_rotmat: 目标旋转矩阵,3x3 的 nparray,可以是单一值或列表
        :param seed_jnt_values: 数值迭代中的起始配置
        :param tcp_jnt_id: 末端执行器关节ID,位于 self.tgtjnts 中
        :param tcp_loc_pos: 末端执行器位置,1x3 的 nparray,描述在 self.jnts[tcp_jnt_id] 的局部坐标系中,可以是单一值或列表
        :param tcp_loc_rotmat: 末端执行器旋转矩阵,3x3 的 nparray,描述在 self.jnts[tcp_jnt_id] 的局部坐标系中,可以是单一值或列表
        :param max_niter: 最大迭代次数
        :param local_minima: 在局部最小值处的处理方式,"accept"(接受),"randomrestart"(随机重启),"end"(结束)
        :param toggle_debug: 是否启用调试模式
        :return: 求解出的逆向运动学配置
        """
        return self._ikt.num_ik(tgt_pos=tgt_pos,
                                tgt_rot=tgt_rotmat,
                                seed_jnt_values=seed_jnt_values,
                                max_niter=max_niter,
                                tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                local_minima=local_minima,
                                toggle_debug=toggle_debug)
    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        """
        计算操作性(Manipulability)
        
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置
        :param tcp_loc_rotmat: 末端执行器旋转矩阵
        :return: 操作性值
        """
        tcp_jnt_id = self.tcp_jnt_id if tcp_jnt_id is None else tcp_jnt_id
        tcp_loc_pos = self.tcp_loc_pos if tcp_loc_pos is None else tcp_loc_pos
        tcp_loc_rotmat = self.tcp_loc_rotmat if tcp_loc_rotmat is None else tcp_loc_rotmat
        return self._ikt.manipulability(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat,
                             type="translational"):
        """
        计算操作性(Manipulability)的雅可比矩阵
        
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置
        :param tcp_loc_rotmat: 末端执行器旋转矩阵
        :param type: 计算类型,"translational" 或 "rotational"
        :return: 操作性雅可比矩阵
        """
        return self._ikt.manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                              tcp_loc_pos=tcp_loc_pos,
                                              tcp_loc_rotmat=tcp_loc_rotmat,
                                              type=type)

    def jacobian(self,
                 tcp_jnt_id,
                 tcp_loc_pos,
                 tcp_loc_rotmat):
        """
        计算雅可比矩阵
        
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置
        :param tcp_loc_rotmat: 末端执行器旋转矩阵
        :return: 雅可比矩阵
        """
        tcp_jnt_id = self.tcp_jnt_id if tcp_jnt_id is None else tcp_jnt_id
        tcp_loc_pos = self.tcp_loc_pos if tcp_loc_pos is None else tcp_loc_pos
        tcp_loc_rotmat = self.tcp_loc_rotmat if tcp_loc_rotmat is None else tcp_loc_rotmat
        return self._ikt.jacobian(tcp_jnt_id=tcp_jnt_id,
                                  tcp_loc_pos=tcp_loc_pos,
                                  tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_loc_tcp_to_gl(self,
                          loc_pos=np.zeros(3),
                          loc_rotmat=np.eye(3),
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        给定相对于第 i 个关节链(jntlnk)的相对位置和相对旋转矩阵,获取世界坐标系下的位置和旋转矩阵
        
        :param loc_pos: 1x3 的 nparray,位置
        :param loc_rotmat: 3x3 的 nparray,旋转矩阵
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置(局部坐标系)
        :param tcp_loc_rotmat: 末端执行器旋转矩阵(局部坐标系)
        :return: 世界坐标系下的位置和旋转矩阵
        作者:weiwei
        日期:20190312, 20210609
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.tcp_loc_rotmat
        tcp_gl_pos = self.jnts[tcp_jnt_id]['gl_posq'] + self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gl_rotmat = self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_rotmat)
        gl_pos = tcp_gl_pos + tcp_gl_rotmat.dot(loc_pos)
        gl_rot = tcp_gl_rotmat.dot(loc_rotmat)
        return [gl_pos, gl_rot]

    def cvt_gl_to_loc_tcp(self,
                          gl_pos,
                          gl_rotmat,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        给定世界坐标系下的位置和旋转矩阵,计算相对于第 i 个关节链(jntlnk)的相对位置和相对旋转矩阵

        :param gl_pos: 1x3 的 nparray,世界坐标系中的位置
        :param gl_rotmat: 3x3 的 nparray,世界坐标系中的旋转矩阵
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 1x3 的 nparray,末端执行器在 tcp_jnt_id 坐标系下的局部位置
        :param tcp_loc_rotmat: 3x3 的 nparray,末端执行器在 tcp_jnt_id 坐标系下的局部旋转矩阵
        :return: 末端执行器在第 i 个关节链坐标系下的位置和旋转矩阵
        author: weiwei
        date: 20190312
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.tcp_loc_rotmat
        # 计算末端执行器相对于关节的局部坐标
        tcp_gloc_pos = self.jnts[tcp_jnt_id]['gl_posq'] + self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gloc_rotmat = self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_rotmat)
        # 计算相对位置和旋转矩阵
        loc_pos, loc_romat = rm.rel_pose(tcp_gloc_pos, tcp_gloc_rotmat, gl_pos, gl_rotmat)
        return [loc_pos, loc_romat]

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      rgba=None,
                      name='jlcmesh'):
        """
        生成网格模型

        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置
        :param tcp_loc_rotmat: 末端执行器旋转矩阵
        :param toggle_tcpcs: 是否显示工具坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param rgba: 颜色
        :param name: 模型名称
        :return: 网格模型
        """
        return self._mt.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=toggle_tcpcs,
                                      toggle_jntscs=toggle_jntscs,
                                      name=name, rgba=rgba)

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
                       name='jlcstick'):
        """
        生成棒状模型

        :param rgba: 颜色
        :param thickness: 棒的粗细
        :param joint_ratio: 关节比例
        :param link_ratio: 连杆比例
        :param tcp_jnt_id: 末端执行器关节ID
        :param tcp_loc_pos: 末端执行器位置
        :param tcp_loc_rotmat: 末端执行器旋转矩阵
        :param toggle_tcpcs: 是否显示工具坐标系
        :param toggle_jntscs: 是否显示关节坐标系
        :param toggle_connjnt: 是否显示连接关节
        :param name: 模型名称
        :return: 棒状模型
        """
        return self._mt.gen_stickmodel(rgba=rgba,
                                       thickness=thickness,
                                       joint_ratio=joint_ratio,
                                       link_ratio=link_ratio,
                                       tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=toggle_tcpcs,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt,
                                       name=name)

    def gen_endsphere(self):
        """
        生成末端球体模型

        :return: 末端球体模型
        """
        return self._mt.gen_endsphere()

    def copy(self):
        """
        创建当前对象的深拷贝

        :return: 当前对象的深拷贝
        """
        return copy.deepcopy(self)


if __name__ == "__main__":
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3, 0, 3], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    jlinstance = JLChain(pos=np.array([0, .01, .03]), rotmat=rm.rotmat_from_axangle([1, 0, 0], np.radians(45)),
                         homeconf=np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    # rjlinstance.settcp(tcp_jnt_id=rjlinstance.tgtjnts[-3], tcp_loc_pos=np.array([0,0,30]))
    # jlinstance.jnts[4]['type'] = 'prismatic'
    # jlinstance.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
    # jlinstance.jnts[4]['motion_val'] = .2
    # jlinstance.jnts[4]['rngmax'] = 1
    # jlinstance.jnts[4]['rngmin'] = -1

    jlinstance.fk(np.array([0, 0, 0, 0, np.radians(90), np.radians(30), np.radians(15), np.radians(20)]))
    # print(jlinstance.jnts)
    jlinstance.gen_stickmodel().attach_to(base)
    # base.run()

    tgt_pos0 = np.array([.45, 0, 0])
    tgt_rotmat0 = np.eye(3)
    tgt_pos1 = np.array([.1, 0, 0])
    tgt_rotmat1 = np.eye(3)
    tgt_pos_list = [tgt_pos0, tgt_pos1]
    tgt_rotmat_list = [tgt_rotmat0, tgt_rotmat1]

    gm.gen_mycframe(pos=tgt_pos0, rotmat=tgt_rotmat0, length=.15, thickness=.01).attach_to(base)
    gm.gen_mycframe(pos=tgt_pos1, rotmat=tgt_rotmat1, length=.15, thickness=.01).attach_to(base)

    tcp_jnt_id_list = [jlinstance.tgtjnts[-1], jlinstance.tgtjnts[-6]]
    tcp_loc_poslist = [np.array([.03, 0, .0]), np.array([.03, 0, .0])]
    tcp_loc_rotmatlist = [np.eye(3), np.eye(3)]
    # tgt_pos_list = tgt_pos_list[0]
    # tgt_rotmat_list = tgt_rotmat_list[0]
    # tcp_jnt_id_list = tcp_jnt_id_list[0]
    # tcp_loc_poslist = tcp_loc_poslist[0]
    # tcp_loc_rotmatlist = tcp_loc_rotmatlist[0]

    tic = time.time()
    jnt_values = jlinstance.ik(tgt_pos_list,
                               tgt_rotmat_list,
                               seed_jnt_values=None,
                               tcp_jnt_id=tcp_jnt_id_list,
                               tcp_loc_pos=tcp_loc_poslist,
                               tcp_loc_rotmat=tcp_loc_rotmatlist,
                               local_minima="accept",
                               toggle_debug=True)
    toc = time.time()
    print('ik cost: ', toc - tic, jnt_values)
    jlinstance.fk(jnt_values=jnt_values)
    jlinstance.gen_stickmodel(tcp_jnt_id=tcp_jnt_id_list,
                              tcp_loc_pos=tcp_loc_poslist,
                              tcp_loc_rotmat=tcp_loc_rotmatlist,
                              toggle_jntscs=True).attach_to(base)

    jlinstance2 = jlinstance.copy()
    jlinstance2.fix_to(pos=np.array([1, 1, 0]), rotmat=rm.rotmat_from_axangle([0, 0, 1], math.pi / 2))
    jlinstance2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id_list,
                               tcp_loc_pos=tcp_loc_poslist,
                               tcp_loc_rotmat=tcp_loc_rotmatlist,
                               toggle_jntscs=True).attach_to(base)
    base.run()
