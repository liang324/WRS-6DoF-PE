import math
import numpy as np
import basis.robot_math as rm


class Jnt(object):
    """
    表示机器人中的一个关节

    :param name: 关节的名称,默认为 "joint".
    :param type: 关节的类型,默认为 "revolute"(旋转关节).
    :param loc_pos: 关节的局部位置,默认为 [0, 0, 0.1].
    :param loc_rotmat: 关节的局部旋转矩阵,默认为单位矩阵.
    :param loc_motionax: 关节的局部运动轴,默认为 [0, 0, 1].
    :param gl_pos0: 关节的初始全局位置,将由正向运动学更新.
    :param gl_rotmat0: 关节的初始全局旋转矩阵,将由正向运动学更新.
    :param gl_motionax: 关节的全局运动轴,将由正向运动学更新.
    :param gl_posq: 关节的全局位置,将由正向运动学更新.
    :param gl_rotmatq: 关节的全局旋转矩阵,将由正向运动学更新.
    :param rng_min: 关节的最小运动范围,默认为 -π.
    :param rng_max: 关节的最大运动范围,默认为 +π.
    :param motion_val: 关节当前的运动值,默认为 0.
    :param p_name: 父关节的名称.
    :param chd_name_list: 子关节的名称列表.
    :param lnk_name_dict: 子关节与链节的映射字典.
    """

    def __init__(self, name="joint",
                 type="revolute",
                 loc_pos=np.array([0, 0, .1]),
                 loc_rotmat=np.eye(3),
                 loc_motionax=np.array([0, 0, 1]),
                 gl_pos0=np.zeros(3),  # 将由正向运动学更新
                 gl_rotmat0=np.eye(3),  # 将由正向运动学更新
                 gl_motionax=np.zeros(3),  # 将由正向运动学更新
                 gl_posq=np.zeros(3),  # 将由正向运动学更新
                 gl_rotmatq=np.eye(3),  # 将由正向运动学更新
                 rng_min=-math.pi,
                 rng_max=+math.pi,
                 motion_val=.0,
                 p_name=None,
                 chd_name_list=[],
                 lnk_name_dict={}):
        self.name = name
        self.type = type
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motionax = loc_motionax
        self.gl_pos0 = gl_pos0
        self.gl_rotmat0 = gl_rotmat0
        self.gl_motionax = gl_motionax
        self.gl_posq = gl_posq
        self.gl_rotmatq = gl_rotmatq
        self.rng_min = rng_min
        self.rng_max = rng_max
        self.motion_val = motion_val
        self.p_name = p_name  # 父关节名称
        self.chd_name_list = chd_name_list  # 子关节名称列表
        self.lnk_name_dict = lnk_name_dict  # 子关节与链节的映射字典


class Lnk(object):
    """
    表示机器人中的一个链节

    :param name: 链节的名称,默认为 "link"
    :param refjnt_name: 参考关节的名称
    :param loc_pos: 链节的局部位置,默认为 [0, 0, 0]
    :param loc_rotmat: 链节的局部旋转矩阵,默认为单位矩阵
    :param gl_pos: 链节的全局位置,将由正向运动学更新
    :param gl_rotmat: 链节的全局旋转矩阵,将由正向运动学更新
    :param com: 链节的质心位置,默认为 [0, 0, 0]
    :param intertia: 链节的惯性矩阵,默认为单位矩阵
    :param mass: 链节的质量,默认为 0
    :param meshfile: 链节的网格文件
    :param collisionmodel: 链节的碰撞模型
    :param rgba: 链节的颜色,默认为 [0.7, 0.7, 0.7, 1]
    """

    def __init__(self, name="link",
                 refjnt_name=None,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 gl_pos=np.zeros(3),  # to be updated by fk
                 gl_rotmat=np.eye(3),  # to be updated by fk
                 com=np.zeros(3),
                 intertia=np.eye(3),
                 mass=.0,
                 meshfile=None,
                 collisionmodel=None,
                 rgba=np.array([.7, .7, .7, 1])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.gl_pos = gl_pos
        self.gl_rotmat = gl_rotmat
        self.com = com
        self.intertia = intertia
        self.mass = mass
        self.meshfile = meshfile
        self.collisionmodel = collisionmodel
        self.rgba = rgba


class JLTree(object):
    """
    使用 Networkx 的有向图(DiGraph)来定义关节链(Joint Links)

    该类实现了通过关节和链节创建机器人结构的方法,包含初始化关节链、设置机器人关节和链节的相关属性等
    """

    def __init__(self, position=np.zeros(3), rotmat=np.eye(3), initconf=np.zeros(6), name='manipulator'):
        """
        初始化机器人模型

        :param position: 机器人基坐标系的位置,默认为零向量.
        :param rotmat: 机器人基坐标系的旋转矩阵,默认为单位矩阵.
        :param initconf: 机器人的初始关节配置,默认为零向量.
        :param name: 机器人名称,默认为 'manipulator'.
        """
        self.name = name
        self.position = np.array(position)
        self.rotmat = np.array(rotmat)
        self.ndof = initconf.shape[0]  # 机器人的自由度数(关节数量)
        self.jntrng_safemargin = 0  # 关节范围的安全余量(暂时没有使用)
        # 初始化关节和链节集合
        # 一个紧密连接节点的 nx1 列表,包含至少两个 jnt_name
        self.jnt_collection, self.lnk_collection, self._base = self._initjntlnks()
        self.tgtjnt_ids = ['jnt' + str(id) for id in range(self.ndof)]  # 目标关节ID列表

    def _initgraph(self):
        """
        初始化机器人结构图,包括关节和链节
        :return: 返回关节集合、链节集合以及机器人的基节点列表
        """
        # 创建链节集合
        lnk_collection = {}
        lnk_name = self.name + '_lnk_f0j0'
        lnk_collection[lnk_name] = Lnk()  # 创建第一个链节
        for id in range(self.ndof - 1):
            lnk_name = self.name + '_link_j' + str(id) + 'j' + str(id + 1)
            lnk_collection[lnk_name] = Lnk()  # 创建其他链节
        lnk_name = self.name + '_link_j' + str(self.ndof - 1) + 'f1'
        lnk_collection[lnk_name] = Lnk()  # 创建最后一个链节

        # 创建关节集合
        jnt_collection = {}
        jnt_name = self.name + '_f0'
        jnt_collection[jnt_name] = Jnt(type='fixed')  # 创建固定关节
        jnt_collection[jnt_name].p_name = None
        jnt_collection[jnt_name].chd_name_list = [self.name + '_j0']
        jnt_collection[jnt_name].lnk_name_dict[self.name + '_j0'] = [self.name + '_lnk_f0j0']

        # 创建自由度关节
        for id in range(self.ndof):
            jnt_name = self.name + '_j' + str(id)
            jnt_collection[jnt_name] = Jnt()  # 创建每个关节
            if id == 0:
                jnt_collection[jnt_name].p_name = self.name + '_f0'
                jnt_collection[jnt_name].chd_name_list = [self.name + '_j1']
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_j1'] = [self.name + '_lnk_f0j1']
            elif id == self.ndof - 1:
                jnt_collection[jnt_name].p_name = self.name + '_j' + str(id - 1)
                jnt_collection[jnt_name].chd_name_list = [self.name + '_f1']
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_f1'] = [self.name + '_lnk_j' + str(id) + 'f1']
            else:
                jnt_collection[jnt_name].p_name = self.name + '_j' + str(id - 1)
                jnt_collection[jnt_name].chd_name_list = [self.name + '_j' + str(id + 1)]
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_j' + str(id + 1)] = [
                    self.name + '_lnk_j' + str(id) + 'j' + str(id + 1)]

        # 创建最后一个固定关节
        jnt_name = self.name + '_f1'
        jnt_collection[jnt_name] = Jnt(type='fixed')  # 创建固定关节
        jnt_collection[jnt_name].p_name = self.name + '_joint' + str(self.ndof - 1)
        jnt_collection[jnt_name].chd_name_list = []
        jnt_collection[jnt_name].lnk_name_dict = {}
        # 返回关节集合、链节集合以及基节点
        return jnt_collection, lnk_collection, [self.name + '_fixed0', self.name + '_joint0']

    def _update_jnt_fk(self, jnt_name):
        """
        递归地更新关节的正向运动学(FK)树
        :param jnt_name: 需要更新的关节名称
        author: weiwei
        date: 20201204osaka
        """
        p_jnt_name = self.jnt_collection[jnt_name].p_name  # 获取当前关节的父关节名称
        cur_jnt = self.jnt_collection[jnt_name]  # 获取当前关节对象
        # 更新全局位置和全局旋转矩阵
        if p_jnt_name is None:
            # 如果没有父关节,当前关节的位置和旋转矩阵就是它的局部位置和旋转矩阵
            cur_jnt.gl_pos0 = cur_jnt.loc_pos
            cur_jnt.gl_rotmat0 = cur_jnt.loc_rotmat
        else:
            p_jnt = self.jnt_collection[p_jnt_name]  # 获取父关节
            curjnt_loc_pos = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_pos)  # 计算当前关节相对于父关节的位置
            cur_jnt.gl_pos0 = p_jnt.gl_posq + curjnt_loc_pos  # 当前关节的全局位置
            cur_jnt.gl_rotmat0 = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_rotmat)  # 当前关节的全局旋转矩阵
            cur_jnt.gl_motionax = np.dot(cur_jnt.gl_rotmat0, cur_jnt.loc_motionax)  # 当前关节的全局运动轴

        # 更新关节的全局位置和旋转矩阵
        if cur_jnt.type == "dummy":
            cur_jnt.gl_posq = cur_jnt.gl_pos0
            cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
        elif cur_jnt.type == "revolute":
            cur_jnt.gl_posq = cur_jnt.gl_pos0
            curjnt_loc_rotmat = rm.rotmat_from_axangle(cur_jnt.loc_motionax, cur_jnt.motion_val)  # 根据轴角法则生成旋转矩阵
            cur_jnt.gl_rotmatq = np.dot(cur_jnt.gl_rotmat0, curjnt_loc_rotmat)  # 更新全局旋转矩阵
        elif cur_jnt.type == "prismatic":
            cur_jnt.gl_posq = cur_jnt.gl_pos0 + cur_jnt.motion_val * cur_jnt.loc_motionax  # 平移关节
            cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
        else:
            # 如果关节类型不可用,抛出错误
            raise ValueError("The given joint type is not available!")
        # 递归更新子关节
        for each_jnt_name in cur_jnt.chd_name_list:
            self._update_jnt_fk(each_jnt_name)

    def _update_jnt_fk_faster(self, jnt_name):
        """
        判断该方法是否比 `_update_jnt_fk` 更快
        :param jnt_name: 需要更新的关节名称
        author: weiwei
        date: 20201203osaka
        """
        while jnt_name is not None:
            cur_jnt = self.jnt_collection[jnt_name]  # 获取当前关节对象
            p_jnt_name = cur_jnt.p_name  # 获取父关节名称
            # 更新关节的全局位置和旋转矩阵
            if p_jnt_name is None:
                cur_jnt.gl_pos0 = cur_jnt.loc_pos  # 如果没有父关节,直接使用局部位置作为全局位置
                cur_jnt.gl_rotmat0 = cur_jnt.loc_rotmat  # 使用局部旋转矩阵
            else:
                p_jnt = self.jnt_collection[p_jnt_name]  # 获取父关节对象
                curjnt_loc_pos = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_pos)  # 计算当前关节相对于父关节的位置
                cur_jnt.gl_pos0 = p_jnt.gl_posq + curjnt_loc_pos  # 当前关节的全局位置
                cur_jnt.gl_rotmat0 = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_rotmat)  # 当前关节的全局旋转矩阵
                cur_jnt.gl_motionax = np.dot(cur_jnt.gl_rotmat0, cur_jnt.loc_motionax)  # 计算当前关节的全局运动轴

            # 更新关节的全局位置(gl_posq)和旋转矩阵(gl_rotmatq)
            if cur_jnt.type == "dummy":
                cur_jnt.gl_posq = cur_jnt.gl_pos0
                cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
            elif cur_jnt.type == "revolute":
                cur_jnt.gl_posq = cur_jnt.gl_pos0
                curjnt_loc_rotmat = rm.rotmat_from_axangle(cur_jnt.loc_motionax, cur_jnt.motion_val)  # 使用轴角法则生成旋转矩阵
                cur_jnt.gl_rotmatq = np.dot(cur_jnt.gl_rotmat0, curjnt_loc_rotmat)  # 计算当前关节的全局旋转矩阵
            elif cur_jnt.type == "prismatic":
                cur_jnt.gl_posq = cur_jnt.gl_pos0 + cur_jnt.motion_val * cur_jnt.loc_motionax  # 平移关节
                cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
            else:
                raise ValueError("The given joint type is not available!")  # 如果关节类型不支持,抛出错误
            # 递归更新所有子关节
            for each_jnt_name in cur_jnt.chd_name_list:
                self._update_jnt_fk(each_jnt_name)  # 递归调用更新子关节

    def _update_lnk_fk(self, base=None):
        """
        更新链节的正向运动学
        author: weiwei
        date: 20201203osaka
        """
        for lnk_name in self.lnk_collection.keys():  # 遍历所有链节
            cur_lnk = self.lnk_collection[lnk_name]  # 获取当前链节对象
            ref_jnt = self.jnt_collection[cur_lnk.refjnt_name]  # 获取链节参考关节
            # 计算链节的全局位置和旋转矩阵
            cur_lnk.gl_pos = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_pos) + ref_jnt.gl_posq
            cur_lnk.gl_rotmat = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_rotmat)

    def _update_fk(self, base=None):
        """
        更新正向运动学(Forward Kinematics,FK)
        递归地更新所有关节和链节的位置和旋转,依据关节的当前状态

        :param base: 可选参数,指定基准关节列表,默认使用 `self._base`
        """
        if base is None:
            base = self._base  # 如果没有传入 base,则使用当前的 _base
        for jnt_name in base:  # 遍历基准关节列表
            for chd_name in self.jnt_collection[jnt_name].chd_name_list:  # 遍历当前关节的子关节
                if chd_name not in base:
                    self._update_jnt_fk(jnt_name=chd_name)  # 递归地更新子关节
        # 更新链节的正向运动学
        self._update_lnk_fk()

    def fk(self, root=None, jnt_motion_vals=None):
        """
        使用正向运动学(Forward Kinematics)来移动关节
        :param jnt_motion_vals: 一个 nx1 列表,每个元素表示一个关节的运动值(单位可以是弧度或米)；字典格式,{关节名称: 运动值, ...}
        :return: None
        author: weiwei
        date: 20161205, 20201009osaka
        """
        if isinstance(jnt_motion_vals, list):  # 如果运动值是一个列表
            if len(jnt_motion_vals) != len(self.tgtjnt_names):
                raise ValueError("The number of given joint motion values must be coherent with self.tgtjnt_names!")
            counter = 0
            for jnt_name in self.tgtjnt_names:  # 为每个目标关节设置对应的运动值
                self.jnt_collection[jnt_name].motion_val = jnt_motion_vals[counter]
                counter += 1
        if isinstance(jnt_motion_vals, dict):  # 如果运动值是一个字典
            pass
        # 更新正向运动学
        self._update_fk()

    def change_base(self, base):
        """
        两个关节确定一个基准
        注释:
        - 如果一个关节是子关节,其下属的所有家族会向下流动
        - 如果关节有父关节,家族流动的方向就不明确,原始基准可能在某些地方位于其上方
        - 因此,当一个关节列表被设置为新的基准时,它们之间的家族关系会被忽略并重新保持
        - 在另一个关节列表被设置为基准时,它们的家族关系会被更新
        :param base: 一个 n x 1 的关节名称列表,至少包含两个关节名称,表示连接的基准关节.
        :return: None
        author: weiwei
        date: 20201203
        """
        self._base = base  # 设置新的基准关节
        for jnt_name in base:  # 遍历基准关节
            cur_jnt = self.jnt_collection[jnt_name]  # 获取当前关节对象
            cur_jnt.loc_pos = cur_jnt.gl_posq  # 将局部位置设置为全局位置
            cur_jnt.loc_rotmat = cur_jnt.gl_posq  # 将局部旋转矩阵设置为全局旋转矩阵
            # 如果当前关节有父关节,并且父关节不在基准关节列表中
            if cur_jnt.p_name not in base:
                p_jnt_name = cur_jnt.p_name  # 获取父关节的名称
                cur_jnt.p_name = None  # 断开父子关节关系
                while p_jnt_name is not None:
                    # 反转 cur_jnt 的所有父级
                    p_jnt = self.jnt_collection[p_jnt_name]  # 获取父关节对象
                    p_jnt.p_name = jnt_name  # 将父关节的父关节指向当前关节
                    cur_jnt.chd_name_list = cur_jnt.chd_name_list + [p_jnt_name]  # 将父关节加入子关节列表
                    p_jnt.chd_name_list.remove(jnt_name)  # 从父关节的子关节列表中移除当前关节
                    # 反转链节
                    lnk_name = p_jnt.lnk_name_dict[jnt_name]  # 获取链节名称
                    cur_jnt.lnk_name_dict[p_jnt_name] = lnk_name  # 反转链节字典
                    p_lnk = self.lnk_collection[lnk_name]  # 获取链节对象
                    p_lnk.loc_pos = p_jnt.loc_pos + np.dot(p_jnt.loc_rotmat, p_lnk.loc_pos)  # 更新链节位置
                    p_lnk.loc_rotmat = np.dot(p_jnt.loc_rotmat, p_lnk.loc_rotmat)  # 更新链节旋转矩阵

                    # p_lnk.com = com
                    # p_lnk.intertia = intertia
                    # p_lnk.mass = mass

                    # 反转旋转轴
                    p_jnt.loc_motionax = -p_jnt.loc_motionax

                    # 迭代更新
                    p_jnt.lnk_name_dict.pop[jnt_name]  # 从父关节的链节字典中移除当前关节
                    cur_jnt = p_jnt  # 更新当前关节为父关节
                    p_jnt_name = cur_jnt.p_name  # 获取父关节的新名称
                    cur_jnt.p_name = None  # 清除父关节的父关节指针

#
# if __name__ == '__main__':
#     jl = JntLnks()
