import math
import scipy
import operator
import warnings
import functools
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import basis.trimesh.creation as trm_creation

try:
    import basis.robotmath_fast as rmf

    fast_math = True
except:
    fast_math = False

# 用于测试一个数是否接近于零 存储 NumPy float32 类型的最小可表示误差(epsilon)
# _EPS 的值约为 1.1920929e-07,用于判断浮点数是否接近 0,常用于数值计算的稳定性判断.
# _EPS = np.finfo(float).eps
_EPS = np.finfo(float).eps * 4.0

# 欧拉角的轴序列
_NEXT_AXIS = [1, 2, 0, 1]

# 将轴字符串映射到/从内部轴、奇偶、重复、帧的元组
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# rotmat
# @numba.jit(fastmath=True, parallel=True)
if fast_math:
    def rotmat_from_axangle(axis, angle):
        """
        使用给定的轴和角度计算 Rodrigues(罗德里格)旋转矩阵

        :param axis: 1x3 的 numpy 数组,表示旋转轴
        :param angle: 弧度制的旋转角度
        :return: 3x3 的旋转矩阵

        作者: weiwei
        日期: 20161220
        """
        axis = rmf.unit_vector(np.array(axis))  # 将轴归一化
        return rmf.rotmat_from_axangle(axis, angle)  # 使用外部库的函数计算旋转矩阵

else:
    def rotmat_from_axangle(axis, angle):
        """
        使用给定的轴和角度计算 Rodrigues(罗德里格)旋转矩阵

        :param axis: 1x3 的 numpy 数组,表示旋转轴
        :param angle: 弧度制的旋转角度
        :return: 3x3 的旋转矩阵

        作者: weiwei
        日期: 20161220
        """
        axis = unit_vector(np.array(axis))  # 将输入轴转换为单位向量
        a = math.cos(angle / 2.0)  # 四元数的实部
        b, c, d = -axis * math.sin(angle / 2.0)  # 四元数的虚部(乘以负号)

        # 为了构造旋转矩阵,先计算四元数的分量平方和乘积
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        # 构建 3x3 的旋转矩阵
        return np.array([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
                         [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
                         [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc]])


def rotmat_from_quaternion(quaternion):
    """
    将四元数转换为旋转矩阵

    :param quaternion: 四元数,通常为长度为4的数组
    :return: 4x4 的旋转矩阵
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    # 计算四元数的模平方
    n = np.dot(q, q)
    # 如果模平方小于一个非常小的值,则返回单位矩阵
    if n < _EPS:
        return np.identity(4)
    # 归一化四元数
    q *= math.sqrt(2.0 / n)
    # 计算四元数外积
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def rotmat_from_normal(surfacenormal):
    """
    使用表面法线计算 3D 网格的旋转矩阵

    :param surfacenormal: 1x3 的 numpy 数组,表示表面法线
    :return: 3x3 的旋转矩阵

    :date: 20160624
    :author: weiwei
    """
    rotmat = np.eye(3, 3)  # 初始化为单位矩阵
    rotmat[:, 2] = unit_vector(surfacenormal)  # 将法线向量设置为旋转矩阵的第三列
    rotmat[:, 0] = orthogonal_vector(rotmat[:, 2], toggle_unit=True)  # 计算一个与法线正交的向量,并设置为第一列
    rotmat[:, 1] = np.cross(rotmat[:, 2], rotmat[:, 0])  # 计算第二列,确保矩阵正交
    return rotmat


def rotmat_from_normalandpoints(facetnormal, facetfirstpoint, facetsecondpoint):
    '''
    使用法线和前两个点计算 3D 面片的旋转矩阵,该函数使用了 Trimesh 定义的概念

    :param facetnormal: 1x3 的 numpy 数组,表示面片法线
    :param facetfirstpoint: 1x3 的 numpy 数组,表示面片上的第一个点
    :param facetsecondpoint: 1x3 的 numpy 数组,表示面片上的第二个点
    :return: 3x3 的旋转矩阵

    日期: 20160624
    作者: weiwei
    '''
    rotmat = np.eye(3, 3)  # 初始化为单位矩阵
    rotmat[:, 2] = unit_vector(facetnormal)  # 将法线向量设置为旋转矩阵的第三列
    rotmat[:, 0] = unit_vector(facetsecondpoint - facetfirstpoint)  # 计算第一个和第二个点之间的向量,设置为旋转矩阵的第一列
    # 如果计算出来的第一列为零向量,说明提供的两个点相同
    if np.allclose(rotmat[:, 0], 0):
        warnings.warn("The provided facetpoints are the same! An autocomputed vector is used instead...")
        rotmat[:, 0] = orthogonal_vector(rotmat[:, 2], toggle_unit=True)  # 使用一个自动计算的正交向量代替
    # 计算第二列,确保旋转矩阵的正交性
    rotmat[:, 1] = np.cross(rotmat[:, 2], rotmat[:, 0])
    return rotmat


def rotmat_from_euler(ai, aj, ak, axes='sxyz'):
    """
    从欧拉角计算旋转矩阵

    :param ai: 弧度制的第一个欧拉角
    :param aj: 弧度制的第二个欧拉角
    :param ak: 弧度制的第三个欧拉角
    :param axes: 旋转轴的顺序(默认 'sxyz')
    :return: 3x3 旋转矩阵

    作者: weiwei
    日期: 20190504
    """
    # 调用外部函数 _euler_matrix 并返回前三行前三列
    return _euler_matrix(ai, aj, ak, axes)[:3, :3]


def rotmat_to_euler(rotmat, axes='sxyz'):
    """
    从旋转矩阵转换为欧拉角

    :param rotmat: 3x3 的旋转矩阵
    :param axes: 旋转轴的顺序(默认 'sxyz')
    :return: 返回欧拉角(弧度制)

    作者: weiwei
    日期: 20190504
    """
    ax, ay, az = _euler_from_matrix(rotmat, axes)
    return np.array([ax, ay, az])


def rotmat_from_two_axis(a, b, axis="xy"):
    """
    根据两个单位向量生成旋转矩阵

    :param a: 第一个单位向量
    :param b: 第二个单位向量
    :param axis: 指定的轴配置,可以是 "xy", "xz" 或 "yz"
    :return: 旋转矩阵

    author: hu
    date: 20240617
    """
    if axis == "xy":
        x = a
        y = b
        z = np.cross(x, y)
    elif axis == "xz":
        x = a
        z = b
        y = np.cross(z, x)
    else:
        y = a
        z = b
        x = np.cross(y, z)
    rotmat = np.array([[x[0], y[0], z[0]],
                       [x[1], y[1], z[1]],
                       [x[2], y[2], z[2]]])
    return rotmat


def rotmat_between_vectors(v1, v2):
    """
    计算从向量 v1 旋转到向量 v2 所需的旋转矩阵

    :param v1: 1x3 的 numpy 数组
    :param v2: 1x3 的 numpy 数组
    :return: 3x3 的旋转矩阵

    作者: weiwei
    日期: 20191228
    """
    theta = angle_between_vectors(v1, v2)  # 计算两向量之间的夹角
    if np.allclose(theta, 0):
        return np.eye(3)  # 如果角度为0,返回单位矩阵(无需旋转)
    if np.allclose(theta, np.pi):
        # 如果两向量方向相反,选择任意一个与 v1 正交的单位向量作为旋转轴
        return rotmat_from_axangle(orthogonal_vector(v1, toggle_unit=True), theta)
    # 向量叉积作为旋转轴方向
    axis = unit_vector(np.cross(v1, v2))
    return rotmat_from_axangle(axis, theta)  # 利用轴角生成旋转矩阵


def rotmat_average(rotmatlist, bandwidth=10):
    """
    对一组旋转矩阵进行平均(基于四元数)

    :param rotmatlist: 旋转矩阵列表(每个为 3x3 numpy 数组)
    :param bandwidth: 平滑参数(传入给 quaternion_average 的带宽参数)
    :return: 平均后的 3x3 旋转矩阵

    作者: weiwei
    日期: 20190422
    """
    if len(rotmatlist) == 0:
        return False  # 空列表,返回 False
    quaternionlist = []
    for rotmat in rotmatlist:
        quaternionlist.append(quaternion_from_matrix(rotmat))  # 将旋转矩阵转换为四元数
    quatavg = quaternion_average(quaternionlist, bandwidth=bandwidth)  # 对四元数进行平均
    rotmatavg = rotmat_from_quaternion(quatavg)[:3, :3]  # 再将平均后的四元数转换回旋转矩阵
    return rotmatavg


def rotmat_slerp(rotmat0, rotmat1, nval):
    """
    对两个旋转矩阵进行球面线性插值(Slerp),返回一系列中间插值旋转矩阵

    :param rotmat0: 初始旋转矩阵(3x3)
    :param rotmat1: 目标旋转矩阵(3x3)
    :param nval: 插值数量(返回 nval 个旋转矩阵,包括起点和终点)
    :return: 1 x nval 的旋转矩阵列表(numpy 数组),每个都是 3x3 的插值矩阵
    """
    key_rots = R.from_matrix((rotmat0, rotmat1))  # 将两个旋转矩阵转换为 Rotation 对象
    key_times = [0, 1]  # 起始时间戳
    slerp = Slerp(key_times, key_rots)  # 创建 Slerp 插值器
    slerp_times = np.linspace(key_times[0], key_times[1], nval)  # 均匀分布的插值时刻
    interp_rots = slerp(slerp_times)  # 执行插值
    return interp_rots.as_matrix()  # 返回插值结果的旋转矩阵表示


# homogeneous matrix
def homomat_from_posrot(pos=np.zeros(3), rot=np.eye(3)):
    """
    根据位置和平移构建 4x4 的齐次变换矩阵(Homogeneous Transformation Matrix)

    :param pos: 1x3 的位移向量(numpy 数组),表示平移
    :param rot: 3x3 的旋转矩阵(numpy 数组),表示旋转
    :return: 4x4 的齐次矩阵,包含了旋转和平移信息

    作者: weiwei
    日期: 20190313
    """
    homomat = np.eye(4, 4)  # 创建一个单位 4x4 矩阵
    homomat[:3, :3] = rot  # 填充旋转部分
    homomat[:3, 3] = pos  # 填充平移部分
    return homomat


def homomat_from_pos_axanglevec(pos=np.zeros(3), axangle=np.ones(3)):
    """
    根据位置和轴角(旋转轴及旋转角度)构建 4x4 的齐次变换矩阵

    :param pos: 1x3 的位移向量(numpy 数组),表示平移
    :param axanglevec: 1x3 的数组,表示旋转轴的单位向量,长度为旋转角度(弧度)
    :return: 4x4 的齐次变换矩阵,表示旋转和平移的组合

    作者: weiwei
    日期: 20200408
    """
    ax, angle = unit_vector(axangle, toggle_length=True)  # 获取单位向量和角度
    rotmat = rotmat_from_axangle(ax, angle)  # 由轴角生成旋转矩阵
    return homomat_from_posrot(pos, rotmat)  # 使用位置和平移生成齐次变换矩阵


def homomat_inverse(homomat):
    """
    计算齐次变换矩阵的逆

    :param homomat: 4x4 的齐次变换矩阵
    :return: 4x4 的逆齐次变换矩阵

    作者: weiwei
    日期: 20161213
    """
    rotmat = homomat[:3, :3]  # 提取旋转部分
    tranvec = homomat[:3, 3]  # 提取平移部分
    invhomomat = np.eye(4, 4)  # 创建一个单位矩阵
    invhomomat[:3, :3] = np.transpose(rotmat)  # 旋转矩阵的转置作为逆矩阵的旋转部分
    invhomomat[:3, 3] = -np.dot(np.transpose(rotmat), tranvec)  # 平移部分的逆
    return invhomomat


def homomat_transform_points(homomat, points):
    """
    使用齐次矩阵对一个点或一组点进行变换(旋转+平移)

    :param homomat: 4x4 齐次变换矩阵 (numpy array)
    :param points:
        - 单个点:1x3 numpy array
        - 多个点:Nx3 numpy array
    :return:
        - 单个点变换后的坐标:1x3 numpy array
        - 多个点变换后的坐标数组:Nx3 numpy array

    作者: weiwei
    日期: 20161213
    """
    if isinstance(points, list):
        points = np.asarray(points)
    if points.ndim == 1:
        # 单点处理
        homopoint = np.array([points[0], points[1], points[2], 1])
        return np.dot(homomat, homopoint)[:3]
    else:
        # 多点批量处理
        homopcdnp = np.ones((4, points.shape[0]))
        homopcdnp[:3, :] = points.T[:3, :]
        transformed_pointarray = homomat.dot(homopcdnp).T
        return transformed_pointarray[:, :3]


def homomat_average(homomatlist, bandwidth=10):
    """
    对一组 4x4 齐次变换矩阵进行平均,分别对位置和旋转部分单独处理

    :param homomatlist: 列表,元素为4x4的齐次矩阵 (list of numpy array)
    :param bandwidth: 平滑带宽参数,控制均值计算过程(通常用于去噪)
    :return: 平均后的齐次矩阵(4x4 numpy array)

    作者: weiwei
    日期: 20200109
    """
    homomatarray = np.asarray(homomatlist)
    posavg = posvec_average(homomatarray[:, :3, 3], bandwidth)
    rotmatavg = rotmat_average(homomatarray[:, :3, :3], bandwidth)
    return homomat_from_posrot(posavg, rotmatavg)


def interplate_pos_rotmat(start_pos,
                          start_rotmat,
                          goal_pos,
                          goal_rotmat,
                          granularity=.01):
    """
    在起点和终点之间进行位置和旋转矩阵的插值,用于轨迹规划

    :param start_pos: 起点位置(1x3 numpy array)
    :param start_rotmat: 起点旋转矩阵(3x3 numpy array)
    :param goal_pos: 终点位置(1x3 numpy array)
    :param goal_rotmat: 终点旋转矩阵(3x3 numpy array)
    :param granularity: 插值精度,每步大约移动的距离
    :return:
        - pos_list: 插值得到的位置列表 (n x 3 numpy array)
        - rotmat_list: 插值得到的旋转矩阵列表 (n个3x3 numpy array)
    """
    len, vec = unit_vector(start_pos - goal_pos, toggle_length=True)
    nval = math.ceil(len / granularity)
    if nval < 2:
        nval = 2
    pos_list = np.linspace(start_pos, goal_pos, nval)
    rotmat_list = rotmat_slerp(start_rotmat, goal_rotmat, nval)
    return pos_list, rotmat_list


def interplate_pos_rotmat_around_circle(circle_center_pos,
                                        circle_ax,
                                        radius,
                                        start_rotmat,
                                        end_rotmat,
                                        granularity=.01):
    """
    沿给定圆形路径插值生成位置列表与旋转矩阵列表

    :param circle_center_pos: 圆心位置,1x3 numpy 数组
    :param circle_ax: 圆弧法向轴(即绕哪个轴转),1x3 numpy 数组
    :param radius: 圆的半径,单位与位置一致(如米或毫米)
    :param start_rotmat: 起始姿态(3x3旋转矩阵)
    :param end_rotmat: 结束姿态(3x3旋转矩阵)
    :param granularity: 插值精度(两个关键点之间的空间距离,单位与坐标一致)
    :return:
        - pos_list: 插值得到的位置列表(沿圆周分布)
        - rotmat_list: 与位置对应的旋转矩阵列表
    """
    # 生成与圆轴垂直的向量,用作初始点方向
    vec = orthogonal_vector(circle_ax)
    # 计算每一段圆弧长度所对应的角度增量
    granularity_radius = granularity / radius
    nval = math.ceil(np.pi * 2 / granularity_radius)  # 总插值数量,遍一圈
    # 插值旋转矩阵(姿态)
    rotmat_list = rotmat_slerp(start_rotmat, end_rotmat, nval)
    # 插值位置(围绕圆周)
    pos_list = []
    # 将 vec 绕 circle_ax 旋转 angle 得到圆周上的点
    for angle in np.linspace(0, math.pi * 2, nval).tolist():
        pos_list.append(rotmat_from_axangle(circle_ax, angle).dot(vec * radius) + circle_center_pos)
    return pos_list, rotmat_list


# quaternion
def quaternion_from_axangle(angle, axis):
    """
    根据旋转角度和旋转轴生成对应的四元数表示

    :param angle: 旋转角度(单位:弧度)
    :param axis: 旋转轴向量,1x3 numpy 数组(不一定归一化)
    :return: 旋转四元数(4维 numpy 数组,格式为 w, x, y, z)

    作者: weiwei
    日期: 20201113
    """
    # 构造四元数向量的虚部部分(x, y, z)
    quaternion = np.array([0.0, axis[0], axis[1], axis[2]])
    # 对向量部分归一化(如果长度大于极小值 _EPS)
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen  # 虚部 = 轴方向 * sin(θ/2)
    quaternion[0] = math.cos(angle / 2.0)
    return quaternion


def quaternion_average(quaternionlist, bandwidth=10):
    """
    对一组四元数进行加权平均,可选用 MeanShift 降噪(完整版本)

    :param quaternionlist: 四元数列表,形状为 (n, 4)
    :param bandwidth: 降噪的带宽参数(如设置为 None 则不使用 MeanShift)
    :return: 平均后的四元数(长度为 4 的 numpy 数组)

    作者: weiwei
    日期: 20190422
    """
    if len(quaternionlist) == 0:
        return False
    quaternionarray = np.array(quaternionlist)
    # 可选:使用 MeanShift 聚类,仅保留主类簇中的四元数
    if bandwidth is not None:
        anglelist = []
        for quaternion in quaternionlist:
            anglelist.append([quaternion_to_axangle(quaternion)[0]])  # 提取角度值
        mt = cluster.MeanShift(bandwidth=bandwidth)
        quaternionarray = quaternionarray[np.where(mt.fit(anglelist).labels_ == 0)]  # 只保留主类簇中的数据
    nquat = quaternionarray.shape[0]
    weights = [1.0 / nquat] * nquat  # 均匀权重
    # 构建对称累加矩阵
    accummat = np.zeros((4, 4))
    wsum = 0
    for i in range(nquat):
        q = quaternionarray[i, :]
        w_i = weights[i]
        accummat += w_i * (np.outer(q, q))  # 进行 rank-1 更新
        wsum += w_i
    # 归一化
    accummat /= wsum
    # 提取最大特征值对应的特征向量,即为平均四元数
    quatavg = np.linalg.eigh(accummat)[1][:, -1]
    return quatavg


def quaternion_to_euler(quaternion, axes='sxyz'):
    """
    将四元数转换为欧拉角

    :param quaternion: 旋转四元数(长度为 4 的 numpy 数组)
    :param axes: 欧拉角的旋转轴顺序(默认 'sxyz')
    :return: 欧拉角,单位为弧度(返回 3 个值)

    作者: weiwei
    日期: 20190504
    """
    # 将四元数转换为旋转矩阵后,再转为欧拉角
    return rotmat_to_euler(rotmat_from_quaternion(quaternion), axes)


def skew_symmetric(posvec):
    """
    计算给定向量的反对称矩阵(用于叉乘操作)

    :param posvec: 三维向量(1x3 的 numpy 数组)
    :return: 对应的 3x3 反对称矩阵

    作者: weiwei
    日期: 20170421
    """
    return np.array([[0, -posvec[2], posvec[1]],
                     [posvec[2], 0, -posvec[0]],
                     [-posvec[1], posvec[0], 0]])


def orthogonal_vector(basevec, toggle_unit=True):
    """
    给定一个向量 np.array([a,b,c]),该函数计算出与之正交的向量,
    使用 np.array([b-c, -a+c, a-c]),然后如果 toggle_unit 为 True,将其归一化.

    :param basevec: 1x3 的 numpy 数组,表示基准向量
    :param toggle_unit: 是否将结果向量归一化,默认为 True
    :return: 1x3 的单位向量(如果 toggle_unit 为 True),否则为正交向量

    作者: weiwei
    日期: 20200528
    """
    a = basevec[0]
    b = basevec[1]
    c = basevec[2]
    if toggle_unit:
        # 如果需要单位化,调用 unit_vector 函数
        return unit_vector(np.array([b - c, -a + c, a - b]))
    else:
        return np.array([b - c, -a + c, a - b])


def rel_pose(pos0, rot0, pos1, rot1):
    """
    计算相对于给定旋转矩阵和位移的相对位姿,给定两个位置和旋转矩阵,返回它们的相对位置和相对旋转

    :param pos0: 1x3 的 numpy 数组,表示第一个位置
    :param rot0: 3x3 的 numpy 数组,表示第一个旋转矩阵
    :param pos1: 1x3 的 numpy 数组,表示第二个位置
    :param rot1: 3x3 的 numpy 数组,表示第二个旋转矩阵
    :return: 相对位移和相对旋转矩阵

    作者: weiwei
    日期: 20180811
    """
    relpos = np.dot(rot0.T, (pos1 - pos0))  # 计算相对位置
    relrot = np.dot(rot0.T, rot1)  # 计算相对旋转矩阵
    return relpos, relrot


def regulate_angle(lowerbound, upperbound, jntangles):
    """
    将关节角度的范围调整为 [lowerbound, upperbound]
    注意:upperbound-lowerbound 必须是 2*np.pi 或 360 的倍数

    :param lowerbound: 最小角度
    :param upperbound: 最大角度
    :param jntangles: 关节角度,可以是单个角度值或角度数组
    :return: 调整后的关节角度
    """
    if isinstance(jntangles, np.ndarray):
        rng = upperbound - lowerbound  # 计算角度范围
        if rng >= 2 * np.pi:  # 如果角度范围大于或等于 360°
            # 对小于下限的角度进行模运算,使其在给定范围内
            jntangles[jntangles < lowerbound] = jntangles[jntangles < lowerbound] % -rng + rng
            jntangles[jntangles > upperbound] = jntangles[jntangles > upperbound] % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles
    else:
        rng = upperbound - lowerbound
        if rng >= 2 * np.pi:
            # 对小于下限的角度进行模运算
            jntangles = jntangles % -rng + rng if jntangles < lowerbound else jntangles % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles


# try:
#     import numba
#
#
#     @numba.njit([numba.float64[:](numba.float64[:]),numba.int32[:][numba.int32[:]]],fastmath=True)
#     def unit_vector(vector):
#         """
#         :param vector: 1-by-3 nparray
#         :return: the unit of a vector
#         author: weiwei
#         date: 20200701osaka
#         """
#         # length = (vector ** 2).sum() ** 0.5
#         length = np.linalg.norm(vector)
#         # if math.isclose(length, 0):
#         if np.abs(length) < 1e-9:
#             return np.zeros_like(vector).astype(np.float)
#         else:
#             return vector / length
# except:


def unit_vector(vector, toggle_length=False):
    """
    计算并返回给定向量的单位向量.如果指定了 `toggle_length=True`,还会返回该向量的长度

    :param vector: 1x3 的 numpy 数组,表示输入向量
    :param toggle_length: 如果为 True,返回向量的长度；否则仅返回单位向量
    :return: 单位向量(如果 toggle_length=False),或者长度和单位向量(如果 toggle_length=True)

    作者: weiwei
    日期: 20200701
    """
    # length = math.sqrt((vector ** 2).sum())  # 计算向量的长度
    length = np.linalg.norm(vector)
    if math.isclose(length, 0):  # 如果长度接近零
        if toggle_length:
            return 0.0, np.zeros_like(vector)  # 如果需要返回长度和单位向量,返回零向量
        else:
            return np.zeros_like(vector)  # 如果只需要单位向量,返回零向量
    if toggle_length:
        return length, vector / length  # 返回长度和单位向量
    else:
        return vector / length  # 返回单位向量


def angle_between_vectors(v1, v2):
    """
    计算两个三维向量之间的夹角,返回的角度单位是弧度

    :param v1: 1x3 的 numpy 数组,表示第一个向量
    :param v2: 1x3 的 numpy 数组,表示第二个向量
    :return: 两个向量之间的夹角,单位为弧度.如果向量零,则返回 None

    作者: weiwei
    日期: 20190504
    """
    l1, v1_u = unit_vector(v1, toggle_length=True)  # 获取第一个向量的单位向量及其长度
    l2, v2_u = unit_vector(v2, toggle_length=True)  # 获取第二个向量的单位向量及其长度
    if l1 == 0 or l2 == 0:  # 如果任意一个向量的长度为零,返回 None
        return None
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))  # 计算夹角的弧度并返回


def angle_between_2d_vectors(v1, v2):
    """
    计算二维向量 v1 到 v2 的夹角,并返回夹角的符号

    :param v1: 2D 向量
    :param v2: 2D 向量
    :return: v1 到 v2 的夹角,单位为弧度,并带有符号

    作者: weiwei
    日期: 20210530
    """
    return math.atan2(v2[1] * v1[0] - v2[0] * v1[1], v2[0] * v1[0] + v2[1] * v1[1])  # 计算并返回带符号的夹角


def deltaw_between_rotmat(rotmati, rotmatj):
    """
    计算从 rotmati 到 rotmatj 的旋转向量(角度乘以轴向量)

    满足关系:rotmat_from_axangle(norm(deltaw), unit_vector(deltaw)) @ rotmati = rotmatj
    :param rotmati: 3x3 的旋转矩阵(初始)
    :param rotmatj: 3x3 的旋转矩阵(目标)
    :return: deltaw,旋转向量(角度 * 单位轴)

    作者: weiwei
    日期: 20200326
    """
    deltarot = np.dot(rotmatj, rotmati.T)  # 差异旋转矩阵
    tempvec = np.array(
        [deltarot[2, 1] - deltarot[1, 2], deltarot[0, 2] - deltarot[2, 0], deltarot[1, 0] - deltarot[0, 1]])
    tempveclength = np.linalg.norm(tempvec)
    if tempveclength > 1e-6:
        deltaw = math.atan2(tempveclength, np.trace(deltarot) - 1.0) / tempveclength * tempvec
    elif deltarot[0, 0] > 0 and deltarot[1, 1] > 0 and deltarot[2, 2] > 0:
        deltaw = np.array([0, 0, 0])  # 基本没有旋转
    else:
        deltaw = np.pi / 2 * (np.diag(deltarot) + 1)  # 特殊情况估算旋转
    return deltaw


def cosine_between_vector(v1, v2):
    """
    计算两个向量之间的夹角余弦值(点积).适用于非单位向量.

    :param v1: 第一个向量(1x3 numpy array)
    :param v2: 第二个向量(1x3 numpy array)
    :return: 点积结果(范围 [-1, 1])
    """
    l1, v1_u = unit_vector(v1, toggle_length=True)
    l2, v2_u = unit_vector(v2, toggle_length=True)
    if l1 == 0 or l2 == 0:
        raise Exception("One of the given vector is [0,0,0].")
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def axangle_between_rotmat(rotmati, rotmatj):
    """
    计算从 rotmati 旋转到 rotmatj 所需的旋转轴(axis)和旋转角(angle).

    :param rotmati: 3x3 初始旋转矩阵
    :param rotmatj: 3x3 目标旋转矩阵
    :return: axis(单位向量),angle(弧度制)
    """
    deltaw = deltaw_between_rotmat(rotmati, rotmatj)
    angle = np.linalg.norm(deltaw)
    ax = deltaw / angle if isinstance(deltaw, np.ndarray) else None
    return ax, angle


def quaternion_to_axangle(quaternion):
    """
    将四元数转换为轴-角表示

    :param quaternion: 四元数(长度为4,w在第0位)
    :return: (angle, axis)——旋转角(弧度)和旋转轴(单位向量)

    作者: weiwei
    日期: 20190421
    """
    lim = 1e-12
    norm = np.linalg.norm(quaternion)
    angle = 0
    axis = [0, 0, 0]
    if norm > lim:
        w = quaternion[0]
        vec = quaternion[1:]
        normvec = np.linalg.norm(vec)
        angle = 2 * math.acos(w)
        axis = vec / normvec
    return angle, axis


def posvec_average(posveclist, bandwidth=10):
    """
    计算一组位置向量(1x3)的平均值

    :param posveclist: 位置向量列表,每个向量是一个 1x3 的 numpy 数组
    :param bandwidth: 如果设置为 True,会应用 MeanShift 去噪
    :return: 平均位置向量

    作者: weiwei
    日期: 20190422
    """
    if len(posveclist) == 0:
        return False
    if bandwidth is not None:
        # 使用 MeanShift 算法去噪,并返回聚类中心作为位置平均
        mt = cluster.MeanShift(bandwidth=bandwidth)
        posvecavg = mt.fit(posveclist).cluster_centers_[0]
        return posvecavg
    else:
        # 无去噪时直接计算平均值
        return np.array(posveclist).mean(axis=0)


def gen_icohomomats(icolevel=1, rotagls=np.linspace(0, 2 * math.pi, 8, endpoint=False), toggleflat=False):
    """
    使用二十面体生成齐次变换矩阵,通过旋转角度变换每个原点-顶点向量

    :param icolevel: 默认值为 1,表示 42 个顶点
    :param position: 齐次变换矩阵的平移部分,默认为原点 [0, 0, 0]
    :param rotagls: 旋转角度列表,默认为 8 个方向
    :param toggleflat: 如果为 True,则将所有矩阵扁平化为单一列表
    :return: [[homomat, ...], ...],每个内嵌列表的大小是角度数目

    作者: weiwei
    日期: 20200701osaka
    """
    returnlist = []
    icos = trm_creation.icosphere(icolevel)  # 创建二十面体(或 icosphere)
    for vert in icos.vertices:
        z = -vert  # 将顶点朝向反转作为旋转轴
        x = orthogonal_vector(z)  # 计算与 z 垂直的向量 x
        y = unit_vector(np.cross(z, x))  # 计算与 z 和 x 垂直的向量 y
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        returnlist.append([])
        for angle in rotagls:
            # 为每个角度生成齐次变换矩阵
            returnlist[-1].append(np.dot(rotmat_from_axangle(z, angle), temprotmat))
    if toggleflat:
        # 如果需要将矩阵扁平化
        return functools.reduce(operator.iconcat, returnlist, [])
    return returnlist


def gen_icohomomats(icolevel=1,
                    position=np.array([0, 0, 0]),
                    rotagls=np.linspace(0, 2 * math.pi, 8,
                                        endpoint=False),
                    toggleflat=False):
    """
    使用二十面体生成旋转矩阵,通过旋转角度变换每个原点-顶点向量

    :param icolevel: 默认值为 1,表示 42 个顶点
    :param rotagls: 旋转角度列表,默认为 8 个方向
    :param toggleflat: 如果为 True,则将所有矩阵扁平化为单一列表
    :return: [[rotmat3, ...], ...],每个内嵌列表的大小是角度数目

    作者: weiwei
    日期: 20191015osaka
    """
    returnlist = []
    icos = trm_creation.icosphere(icolevel)  # 创建二十面体(或 icosphere)
    for vert in icos.vertices:
        z = -vert  # 将顶点朝向反转作为旋转轴
        x = orthogonal_vector(z)  # 计算与 z 垂直的向量 x
        y = unit_vector(np.cross(z, x))  # 计算与 z 和 x 垂直的向量 y
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        returnlist.append([])
        for angle in rotagls:
            # 为每个角度生成旋转矩阵
            tmphomomat = np.eye(4)
            tmphomomat[:3, :3] = np.dot(rotmat_from_axangle(z, angle), temprotmat)
            tmphomomat[:3, 3] = position
            returnlist[-1].append(tmphomomat)
    if toggleflat:
        return functools.reduce(operator.iconcat, returnlist, [])
    return returnlist


def getaabb(pointsarray):
    """
    获取一个 n x 3 点云数组的轴对齐包围盒 (AABB)

    :param pointsarray: n x 3 数组,表示一组三维坐标点
    :return: 包围盒中心点和边界范围 [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    作者: weiwei
    日期: 20191229
    """
    xmax = np.max(pointsarray[:, 0])
    xmin = np.min(pointsarray[:, 0])
    ymax = np.max(pointsarray[:, 1])
    ymin = np.min(pointsarray[:, 1])
    zmax = np.max(pointsarray[:, 2])
    zmin = np.min(pointsarray[:, 2])
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    # volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    return [center, np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])]


def compute_pca(nparray):
    """
    计算主成分分析 (PCA),返回特征值和特征向量矩阵.

    :param nparray: n x d 数组,d 是数据的维度
    :return: 特征值和特征向量矩阵,返回的是特征值 pcv 和特征向量组成的矩阵 axmat,axmat 的每一列是一个特征向量

    作者: weiwei
    日期: 20200701osaka
    """
    ca = np.cov(nparray, y=None, rowvar=False, bias=True)  # rowvar=False: 每列代表一个数据点,bias=True: 偏置的协方差
    pcv, pcaxmat = np.linalg.eig(ca)
    return pcv, pcaxmat


def transform_data_pcv(data, random_rot=True):
    """
    使用主成分分析 (PCA) 进行数据变换,并可选地进行随机旋转

    :param data: 输入数据,n x d 数组
    :param random_rot: 是否进行随机旋转,默认 True
    :return: 变换后的数据和旋转矩阵

    作者: reuishuang
    日期: 20210706
    """
    pcv, pcaxmat = compute_pca(data)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x_v = pcaxmat[:, inx[2]]
    y_v = pcaxmat[:, inx[1]]
    z_v = pcaxmat[:, inx[0]]
    pcaxmat = np.asarray([y_v, x_v, -z_v]).T
    if random_rot:
        pcaxmat = np.dot(rotmat_from_axangle([1, 0, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 1, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 0, 1], math.radians(5)), pcaxmat)
    transformed_data = np.dot(pcaxmat.T, data.T).T
    return transformed_data, pcaxmat


def fit_plane(points):
    """
    拟合一组三维点的最佳拟合平面

    :param points: nx3 的 numpy 数组,表示 n 个三维点
    :return: 平面中心点(几何中心)和单位法向量
    """
    # 计算所有点的平均值(中心点)
    plane_center = points.mean(axis=0)
    # 对偏移后的点阵执行 SVD 分解
    result = np.linalg.svd(points - plane_center)
    # 计算平面法向量(取两个奇异向量叉积,归一化)
    plane_normal = unit_vector(np.cross(result[2][0], result[2][1]))
    return plane_center, plane_normal


def project_to_plane(point, plane_center, plane_normal):
    """
    将一个三维点投影到指定平面上

    :param point: 1x3 三维点
    :param plane_center: 平面中心点
    :param plane_normal: 平面法向量(单位向量)
    :return: 投影后的三维点
    """
    # 计算点到平面的距离(绝对值)
    dist = abs((point - plane_center).dot(plane_normal))
    # 如果在法向量反方向上,反转法向量
    # print((point - plane_center).dot(plane_normal))
    if (point - plane_center).dot(plane_normal) < 0:
        plane_normal = - plane_normal
    # 投影点 = 原始点 - 垂直分量
    projected_point = point - dist * plane_normal
    return projected_point


def points_obb(pointsarray, toggledebug=False):
    """
    计算一组二维或三维点的 OBB(有向最小外包盒)

    :param pointsarray: 输入点集,形状为 nx2 或 nx3 的 numpy 数组
    :param toggledebug: 是否启用调试绘图(二维)
    :return: 返回中心点、包围盒的角点、以及主方向坐标轴组成的旋转矩阵

    作者: weiwei
    日期: 20191229, 20200701osaka
    """
    # 使用 PCA 计算特征向量矩阵(主方向轴)
    pcv, pcaxmat = compute_pca(pointsarray)
    pcaxmat_t = pcaxmat.T
    # 将点集旋转,使其主方向对齐到 x, y, z 轴
    ar = np.dot(pointsarray, np.linalg.inv(pcaxmat_t))
    # 获取变换后点集的最大/最小值(AABB)
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    diff = (maxa - mina) * 0.5
    # 计算包围盒中心
    center = mina + diff
    # 计算角点(2D 或 3D)
    if pointsarray.shape[1] == 2:
        corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]],
                            center + [diff[0], diff[1]], center + [-diff[0], diff[1]]])
    elif pointsarray.shape[1] == 3:
        corners = np.array([center + [-diff[0], -diff[1], -diff[2]], center + [diff[0], -diff[1], -diff[2]],
                            center + [diff[0], diff[1], -diff[2]], center + [-diff[0], diff[1], -diff[2]],
                            center + [-diff[0], diff[1], diff[2]], center + [-diff[0], -diff[1], diff[2]],
                            center + [diff[0], -diff[1], diff[2]], center + [diff[0], diff[1], diff[2]]])
    # 角点和中心点旋转回原始方向
    corners = np.dot(corners, pcaxmat_t)
    center = np.dot(center, pcaxmat_t)
    # 如果启用调试,可视化绘图(仅支持二维)
    if toggledebug:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.scatter(pointsarray[:, 0], pointsarray[:, 1])
        ax.scatter([center[0]], [center[1]])
        ax.plot(corners[:, 0], corners[:, 1], '-')
        plt.axis('equal')
        plt.show()
    return [center, corners, pcaxmat]


def gaussian_ellipsoid(pointsarray):
    """
    计算给定点集的 95% 概率椭球体的轴矩阵

    :param pointsarray: 输入的点集,形状为 nx3 的 numpy 数组
    :return: 中心点和椭球体的轴矩阵

    作者: weiwei
    日期: 20200701
    """
    # 计算点集的主成分分析(PCA)
    pcv, pcaxmat = compute_pca(pointsarray)
    # 计算点集的几何中心(平均点)
    center = np.mean(pointsarray, axis=0)
    axmat = np.eye(3)
    # 使用 95% 概率椭球体的常数(5.991 是卡方分布的临界值)
    axmat[:, 0] = 2 * math.sqrt(5.991 * pcv[0]) * pcaxmat[:, 0]
    axmat[:, 1] = 2 * math.sqrt(5.991 * pcv[1]) * pcaxmat[:, 1]
    axmat[:, 2] = 2 * math.sqrt(5.991 * pcv[2]) * pcaxmat[:, 2]
    return center, axmat


def random_rgba(toggle_alpha_random=False):
    """
    生成一个随机的 1x4 RGBA 颜色列表,范围为 [0, 1]

    :param toggle_alpha_random: 如果为 False,则 alpha 通道为 1；如果为 True,则 alpha 随机变化
    :return: 随机生成的 RGBA 颜色列表
    """
    if not toggle_alpha_random:
        return np.random.random_sample(3).tolist() + [1]  # 不随机透明度,设置为 1
    else:
        return np.random.random_sample(4).tolist()  # 随机生成 RGBA


def get_rgba_from_cmap(id, cm_name='tab20', step=20):
    """
    从 matplotlib 色图 "tab20" 中获取 RGBA 值

    :param id: 颜色的索引
    :param cm_name: 色图名称,默认为 'tab20',可以参考 matplotlib 的色图教程
    :param step: 色图中的步长,用于决定颜色的分布
    :return: 对应 id 的 RGBA 颜色值

    作者: weiwei
    日期: 20210505, 20220404
    """
    cm = plt.get_cmap(cm_name)  # 获取指定的色图
    return list(cm(id % step))  # 获取对应 id 的 RGBA 值


def consecutive(nparray1d, stepsize=1):
    """
    查找数组中的连续序列

    示例:
    a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    consecutive(a)
    返回 [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
    :param nparray1d: 输入的一维数组
    :param stepsize: 步长,默认值为 1
    :return: 连续序列的列表
    """
    # 使用 np.diff 查找不连续的地方,然后分割数组
    return np.split(nparray1d, np.where(np.diff(nparray1d) != stepsize)[0] + 1)


def null_space(npmat):
    """
    计算给定矩阵的零空间

    :param npmat: 输入的矩阵,形状为 nxm 的 numpy 数组
    :return: 零空间的基底矩阵
    """
    return scipy.linalg.null_space(npmat)


def reflection_matrix(point, normal):
    """
    构造一个 4x4 齐次矩阵,用于在给定点 `point` 和法向量 `normal` 所定义的平面上做镜像变换

    :param point: np.array(4,) 齐次坐标中的一个点,表示镜像平面上的某个点
    :param normal: np.array(3,) 三维向量,表示镜像平面的法向量
    :return: 4x4 的反射变换矩阵
    """
    normal = _unit_vector(normal[:3])  # 单位化法向量
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)  # 构造线性部分(旋转+翻转)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal  # 平移部分
    return M


def reflection_from_matrix(matrix):
    """
    给定一个反射矩阵,提取出其对应的镜像平面的点和法向量

    :param matrix: 4x4 的反射矩阵
    :return: tuple (point, normal),point 是平面上的一个点(齐次坐标),normal 是法向量
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    # 法向量:对应特征值 -1 的单位特征向量
    w, V = np.linalg.eig(M[:3, :3])
    i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(V[:, i[0]]).squeeze()

    # 平面上的点:对应特征值 1 的齐次坐标向量
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]  # 还原成真实空间坐标
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """
    生成一个旋转矩阵,该矩阵使得点绕指定轴旋转,轴由点和方向向量定义

    :param angle: 旋转角度(弧度制)
    :param direction: 旋转轴的单位方向向量
    :param point: 旋转的中心点,默认为原点
    :return: 4x4 的旋转矩阵
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = _unit_vector(direction[:3])  # 单位化旋转轴
    # 基于单位向量的旋转矩阵
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # 如果旋转中心不是原点
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """
    从旋转矩阵中提取旋转角度和旋转轴

    :param matrix: 4x4 旋转矩阵
    :return: (angle, direction, point) 旋转角度、旋转轴和旋转中心
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: 对应特征值 1 的单位特征向量
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: 对应特征值 1 的单位特征向量
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    # 根据旋转轴计算旋转角度
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
    """
    返回一个缩放矩阵,通过因子(`factor`)在指定的原点(`origin`)和方向(`direction`)进行缩放
    你可以使用因子为 -1 来实现点对称

    :param factor: 缩放因子
    :param origin: 缩放操作的原点,默认为 None
    :param direction: 缩放操作的方向,默认为 None(均匀缩放)
    :return: 4x4 缩放矩阵
    """
    if direction is None:
        # 均匀缩放
        M = np.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # 非均匀缩放
        direction = _unit_vector(direction[:3])  # 单位化方向向量
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """
    从缩放矩阵中提取缩放因子、原点和方向

    :param matrix: 4x4 缩放矩阵
    :return: 缩放因子、原点和方向
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    factor = np.trace(M33) - 2.0
    try:
        # direction: 对应特征值为 factor 的单位特征向量
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w) - factor) < 1e-8)[0][0]
        direction = np.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # 均匀缩放
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: 对应特征值为 1 的特征向量
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = np.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
    """
    生成一个用于将点投影到平面上的 4x4 矩阵.这个平面由一个点(point)和一个法向量(normal)定义

    你可以选择:
    - 使用方向向量 `direction` 进行平行投影
    - 使用透视点 `perspective` 进行透视投影
    - 若都不提供,则进行正交投影(沿着法线投影)
    如果设置 `pseudo=True`,则在透视投影中将保留相对深度信息(使得 P_perspective = P_orthogonal × P_pseudo)

    :param point: 投影平面上的一点 (3D 坐标)
    :param normal: 投影平面的法向量 (3D 向量)
    :param direction: 用于平行投影的方向向量(可选)
    :param perspective: 用于透视投影的视点(可选)
    :param pseudo: 是否保留透视深度的伪投影信息
    :return: 4x4 投影矩阵(numpy.ndarray)
    """
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = _unit_vector(normal[:3])  # 单位化法向量

    if perspective is not None:
        # 🔭 透视投影
        perspective = np.array(perspective[:3], dtype=np.float64, copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # 保留相对深度:使用伪投影
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # 📐 平行投影
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # ➖ 正交投影
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """
    从 4x4 投影矩阵中提取出其对应的投影平面信息和透视参数

    返回值与 `projection_matrix()` 函数中的参数一一对应,包括:
    - 投影平面上的点 point
    - 法向量 normal
    - 投影方向 direction(仅用于平行投影)
    - 透视点 perspective(仅用于透视投影)
    - 是否是伪透视 pseudo

    :param matrix: 4x4 投影矩阵
    :param pseudo: 若为 True,表示矩阵中包含伪透视信息
    :return: (point, normal, direction, perspective, pseudo)
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]

    # 提取矩阵的特征值和特征向量
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # 平行/正交投影情况
        # point: 特征值为 1 的特征向量
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]

        # direction: M33 中特征值为 0 的单位特征向量
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector corresponding to eigenvalue 0")
        direction = np.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)

        # normal: M33.T 的特征值为 0 的单位向量
        w, V = np.linalg.eig(M33.T)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if len(i):
            # 平行投影
            normal = np.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # 正交投影(此时 direction 与 normal 相同)
            return point, direction, None, None, False
    else:
        # 透视投影情况
        i = np.where(abs(np.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError(
                "no eigenvector not corresponding to eigenvalue 0")
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]

        # normal: 投影平面的法向量
        normal = - M[3, :3]

        # perspective: 透视点位置
        perspective = M[:3, 3] / np.dot(point[:3], normal)
        if pseudo:
            perspective -= normal  # 还原伪透视的视点
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """
    生成用于从视锥体到标准化设备坐标系的变换矩阵

    视锥体的边界是沿着 x(left, right),y(bottom, top),z(near, far)轴对齐的
    如果坐标位于视锥体内,则标准化设备坐标范围为 [-1, 1]
    如果 `perspective` 为 True,视锥体是一个截头金字塔,透视点位于原点,方向沿着 z 轴；
    否则,它是一个正交的标准视图体积(一个盒子)
    透视剪裁矩阵变换的齐次坐标需要被去同质化(除以 w 坐标)

    :param left: 视锥体左边界
    :param right: 视锥体右边界
    :param bottom: 视锥体底部边界
    :param top: 视锥体顶部边界
    :param near: 视锥体近边界
    :param far: 视锥体远边界
    :param perspective: 如果为 True,生成透视投影矩阵,否则生成正交投影矩阵
    :return: 投影剪裁矩阵
    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustum")
    if perspective:
        if near <= _EPS:
            raise ValueError("invalid frustum: near <= 0")
        t = 2.0 * near
        M = [[t / (left - right), 0.0, (right + left) / (right - left), 0.0],
             [0.0, t / (bottom - top), (top + bottom) / (top - bottom), 0.0],
             [0.0, 0.0, (far + near) / (near - far), t * far / (far - near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)],
             [0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)],
             [0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)],
             [0.0, 0.0, 0.0, 1.0]]
    return np.array(M)


def shear_matrix(angle, direction, point, normal):
    """
    返回一个沿着给定方向向量在剪切平面上的剪切矩阵

    剪切平面由一个点和法向量定义.方向向量必须与剪切平面的法向量正交
    一个点 P 被剪切矩阵转换为 P',使得向量 P-P' 与方向向量平行,其长度由角度 P-P'-P" 决定,其中 P' 是点 P 在剪切平面上的正交投影

    :param angle: 剪切角度
    :param direction: 剪切方向向量
    :param point: 剪切平面上的一个点
    :param normal: 剪切平面的法向量
    :return: 剪切矩阵
    """
    normal = _unit_vector(normal[:3])
    direction = _unit_vector(direction[:3])
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError("direction and normal vectors are not orthogonal")
    angle = math.tan(angle)
    M = np.identity(4)
    M[:3, :3] += angle * np.outer(direction, normal)
    M[:3, 3] = -angle * np.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """
    返回剪切角度、方向和剪切平面

    这个函数从给定的剪切矩阵中恢复剪切角度、方向和剪切平面(即法向量)

    :param matrix: 4x4 剪切变换矩阵
    :return: 剪切角度(angle),剪切方向(direction),剪切平面上的点(point)和法向量(normal)
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    # 计算矩阵的特征值和特征向量
    w, V = np.linalg.eig(M33)
    # 找到特征值接近 1 的特征向量
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("no two linear independent eigenvectors found %s" % w)
    # 获取与特征值 1 对应的特征向量
    V = np.real(V[:, i]).squeeze().T
    # 计算法向量:通过计算特征向量的叉积找到法向量
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        w = vector_norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    normal /= lenorm  # 归一化法向量
    # 计算剪切方向和角度
    direction = np.dot(M33 - np.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle  # 归一化剪切方向
    angle = math.atan(angle)  # 计算剪切角度
    # 计算剪切平面上的点:找到对应特征值 1 的特征向量
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()  # 获取剪切平面上的点
    point /= point[3]  # 去同质化处理
    # 返回剪切角度、方向、平面上的点和法向量
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """
    将一个齐次变换矩阵分解为:缩放、剪切、旋转(欧拉角)、平移、透视等变换分量

    参数:
        matrix : array_like
            非退化的齐次变换矩阵(4x4)

    返回:
        scale : 3个方向的缩放因子(x, y, z)
        shear : 剪切因子列表(xy、xz、yz)
        angles : 绕静态 x, y, z 轴的欧拉角(单位:弧度)
        translate : 沿 x, y, z 的平移向量
        perspective : 透视成分的 4 元组表示

    抛出:
        ValueError: 如果矩阵是错误的类型或是退化矩阵
    """
    M = np.array(matrix, dtype=np.float64, copy=True).T  # 转置以适配内部计算
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]  # 归一化最后一个元素为 1

    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0  # 清除投影部分,保留仿射部分
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3,))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    # 处理投影部分
    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))  # 求透视部分
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])  # 无透视

    # 提取平移分量
    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    # 提取旋转、缩放和剪切(从前3行/列)
    row = M[:3, :3].copy()

    # 第一步:提取 x 方向的缩放
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    # 第二步:提取 xy 剪切
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    # 提取 y 方向的缩放
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    # 第三步:提取 xz 和 yz 剪切
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    # 提取 z 缩放
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    # 如果矩阵是左手坐标系(负行列式),则取相反数
    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1
    # 计算欧拉角(ZYX顺序)
    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def compose_matrix(scale=None, shear=None, angles=None, translate=None, perspective=None):
    """
    根据一系列变换生成变换矩阵

    这是 `decompose_matrix` 函数的逆操作

    变换序列包括:
           scale : 包含 3 个缩放因子的向量,用于 x, y, z 轴的缩放
           shear : 包含 x-y、x-z、y-z 轴的剪切因子的列表
           angles : 包含绕 x, y, z 轴旋转的欧拉角列表
           translate : 平移向量,表示沿 x, y, z 轴的平移
           perspective : 透视变换部分的矩阵(通常用于 3D 图形的透视投影)

    返回一个 4x4 的变换矩阵
    """
    # 初始化为单位矩阵
    M = np.identity(4)
    # 透视变换部分 (如果有的话)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]  # 将透视矩阵的前三个元素放入P矩阵
        M = np.dot(M, P)  # 将透视矩阵与当前矩阵相乘

    # 平移变换部分 (如果有的话)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]  # 将平移向量放入平移矩阵
        M = np.dot(M, T)  # 将平移矩阵与当前矩阵相乘

    # 旋转变换部分 (如果有的话)
    if angles is not None:
        R = _euler_matrix(angles[0], angles[1], angles[2], 'sxyz')  # 生成旋转矩阵
        M = np.dot(M, R)  # 将旋转矩阵与当前矩阵相乘

    # 剪切变换部分 (如果有的话)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]  # 设置剪切因子
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)  # 将剪切矩阵与当前矩阵相乘

    # 缩放变换部分 (如果有的话)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]  # 设置缩放因子
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)  # 将缩放矩阵与当前矩阵相乘
    # 最后归一化矩阵
    M /= M[3, 3]  # 将矩阵的每个元素除以最后一个元素,标准化矩阵
    return M


def orthogonalization_matrix(lengths, angles):
    """
    返回晶体学单元格坐标的正交化矩阵

    输入的角度应为弧度制
    返回的矩阵可以将晶胞坐标系转换为笛卡尔坐标系
    其逆矩阵即为“去正交化矩阵”,可用于从笛卡尔坐标系恢复晶胞坐标系
    """
    a, b, c = lengths
    angles = np.radians(angles)  # 将角度从角度制转换为弧度
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    # 计算余弦分量中的矫正项
    co = (cosa * cosb - cosg) / (sina * sinb)
    # 构造正交化矩阵(4x4,齐次坐标)
    return np.array([
        [a * sinb * math.sqrt(1.0 - co * co), 0.0, 0.0, 0.0],
        [-a * sinb * co, b * sina, 0.0, 0.0],
        [a * cosb, b * cosa, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """
    返回一个用于配准两组点的仿射变换矩阵

    v0 和 v1 是形状为 (维度, N) 的数组,至少要包含“维度”个点
    表示原始点集(v0)和目标点集(v1),所有坐标均为非齐次

    参数:
    - shear: 若为 False,仅返回相似变换矩阵；若为 True,包含剪切.
    - scale: 若为 False,仅返回刚性/欧几里得变换矩阵(无缩放).
    - usesvd: 使用 SVD 最小化均方差(RMSD)进行旋转拟合(默认启用).

    默认使用 Hartley 和 Zissermann 的算法.
    若 usesvd=True,则使用 Kabsch 算法(基于 SVD)求最优拟合.
    若 usesvd=False 且维度为3,则使用 Horn 的四元数算法(速度较慢).

    返回的矩阵可以执行旋转、平移和(可选)缩放变换.

    更多示例见 superimposition_matrix()

    """
    # 转换输入为 float64 类型的数组,copy=True 确保原数组不被修改
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]  # 维度,如 2D/3D
    # 检查维度是否合法,并确保两个点集形状一致
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # === 将两个点集的质心平移到原点 ===
    t0 = -np.mean(v0, axis=1)  # v0 的质心
    M0 = np.identity(ndims + 1)  # 创建单位矩阵作为变换矩阵
    M0[:ndims, ndims] = t0  # 设置平移分量
    v0 += t0.reshape(ndims, 1)  # 将 v0 中的点移到以原点为中心

    t1 = -np.mean(v1, axis=1)  # v1 的质心
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    # === 仿射变换路径:允许剪切变换 ===
    if shear:
        # 拼接原始点集和目标点集
        A = np.concatenate((v0, v1), axis=0)
        # SVD 分解,用于构造变换基底
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]  # 原始点集的正交基
        C = vh[ndims:2 * ndims]  # 目标点集的正交基
        t = np.dot(C, np.linalg.pinv(B))  # 求出线性映射矩阵(可能含剪切)
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)  # 扩展列用于齐次坐标
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))  # 拼接成完整的仿射矩阵

    # === 刚性变换路径(不含剪切),通过 SVD 得到旋转矩阵 ===
    elif usesvd or ndims != 3:
        # 协方差矩阵的 SVD
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # 旋转矩阵
        R = np.dot(u, vh)
        # 检查是否为右手系(右手法则)
        if np.linalg.det(R) < 0.0:
            # 若不是,则调整旋转矩阵方向
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # 构造齐次坐标的变换矩阵
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R

    # === 使用四元数(仅限 3D)恢复刚性变换 ===
    else:
        # 构造对称矩阵 N(Horn 四元数算法)
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx + yy + zz, 0.0, 0.0, 0.0],
             [yz - zy, xx - yy - zz, 0.0, 0.0],
             [zx - xz, xy + yx, yy - xx - zz, 0.0],
             [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
        # 取最大特征值对应的特征向量作为四元数
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # 单位化四元数
        # 四元数转换为旋转矩阵(齐次坐标形式)
        M = quaternion_matrix(q)

    # === 如果需要缩放(但不包括剪切) ===
    if scale and not shear:
        # 使用均方和比率计算缩放因子(保持 RMS 比例)
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # === 把质心移回原来的位置(逆向平移) ===
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]  # 归一化齐次矩阵
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """
    返回将给定的 3D 点集转换为第二个点集的变换矩阵

    v0 和 v1 是形状为 (3, *) 或 (4, *) 的数组,至少包含 3 个点
    参数 scale 和 usesvd 的解释请参考更通用的 affine_matrix_from_points 函数
    返回的矩阵是相似变换(similarity)或欧几里得变换矩阵
    此函数在 transformations.c 中有快速的 C 实现

    参数:
    v0 : ndarray
        输入的第一个 3D 点集,形状为 (3, N) 或 (4, N),N 是点的个数.
    v1 : ndarray
        输入的第二个 3D 点集,形状同 v0
    scale : bool, 可选
        是否进行缩放,默认值为 False.设置为 True 时,会计算缩放因子
    usesvd : bool, 可选
        是否使用 SVD 方法进行计算,默认值为 True.如果为 False,将使用其他方法

    返回:
    ndarray
        返回一个 4x4 的变换矩阵,表示相似变换或欧几里得变换

    """
    # 将输入的 v0 和 v1 转换为浮动类型的数组,并截取前 3 行(假设是三维数据)
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
    # 使用 affine_matrix_from_points 函数计算从 v0 到 v1 的仿射变换矩阵
    return affine_matrix_from_points(v0, v1, shear=False, scale=scale, usesvd=usesvd)


def _euler_matrix(ai, aj, ak, axes='sxyz'):
    """
    返回从欧拉角和轴顺序生成的齐次旋转矩阵

    参数:
    ai, aj, ak : 欧拉角的滚转角(roll)、俯仰角(pitch)和偏航角(yaw)
    axes : 字符串或编码元组,表示旋转的轴顺序,共有 24 种轴顺序

    示例:
        R = _euler_matrix(1, 2, 3, 'syxz')
        np.allclose(np.sum(R[0]), -1.34786452)  # 返回 True
        R = _euler_matrix(1, 2, 3, (0, 1, 0, 1))
        np.allclose(np.sum(R[0]), -0.383436184)  # 返回 True
        ai, aj, ak = (4*np.pi) * (np.random.random(3) - 0.5)
        for axes in _AXES2TUPLE.keys():
            R = _euler_matrix(ai, aj, ak, axes)
        for axes in _TUPLE2AXES.keys():
            R = _euler_matrix(ai, aj, ak, axes)

    作者: weiwei
    来源: Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>
    日期: 20200704
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 验证传入的轴顺序是否合法
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai  # 如果坐标系是框架类型,则交换 ai 和 ak
    if parity:
        ai, aj, ak = -ai, -aj, -ak  # 如果是偶数,则取反欧拉角

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)  # 创建 4x4 单位矩阵
    if repetition:
        # 重复旋转矩阵的处理
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        # 非重复旋转矩阵的处理
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def _euler_from_matrix(matrix, axes='sxyz'):
    """
    从旋转矩阵中返回欧拉角,适用于指定的轴顺序.

    参数:
    matrix : 旋转矩阵,形状为 (3, 3) 或 (4, 4)
    axes : 字符串或编码元组,表示旋转的轴顺序(共有 24 种轴顺序)

    注意:多个欧拉角三元组可以描述同一个旋转矩阵.

    示例:
    R0 = _euler_matrix(1, 2, 3, 'syxz')  # 创建一个旋转矩阵
    al, be, ga = _euler_from_matrix(R0, 'syxz')  # 从旋转矩阵中提取欧拉角
    R1 = _euler_matrix(al, be, ga, 'syxz')  # 使用提取的欧拉角生成新的旋转矩阵
    np.allclose(R0, R1) -> True  # 验证 R0 和 R1 是否相同

    angles = (4 * np.pi) * (np.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():  # 遍历所有的轴顺序
        R0 = _euler_matrix(axes=axes, *angles)  # 使用不同的轴顺序生成旋转矩阵
        R1 = _euler_matrix(axes=axes, *_euler_from_matrix(R0, axes))  # 使用欧拉角从矩阵中恢复并生成新的矩阵
        if not np.allclose(R0, R1):  # 检查矩阵是否一致
            print(axes, "failed")

    返回值:
    ax, ay, az : 对应于给定轴顺序的欧拉角(滚转、俯仰、偏航角)
    """
    # 通过 _AXES2TUPLE 映射轴顺序字符串或元组
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 验证输入轴顺序的有效性
        firstaxis, parity, repetition, frame = axes

    # 获取第一个、第二个和第三个旋转轴
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    # 将输入矩阵转换为 3x3 的旋转矩阵
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        # 如果是重复旋转
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            # 检查是否需要进行标准化
            ax = math.atan2(M[i, j], M[i, k])  # 计算滚转角
            ay = math.atan2(sy, M[i, i])  # 计算俯仰角
            az = math.atan2(M[j, i], -M[k, i])  # 计算偏航角
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        # 非重复旋转
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0
    # 如果 parity 为 True,取反欧拉角
    if parity:
        ax, ay, az = -ax, -ay, -az
    # 如果是框架类型的坐标系,交换俯仰角和偏航角
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """
    根据欧拉角(滚转、俯仰、偏航)和轴顺序,返回对应的四元数

    参数:
    ai, aj, ak : 欧拉角 (roll, pitch, yaw)
    axes : 字符串或元组,表示旋转的轴顺序(例如 'sxyz')
    返回:
    q : 四元数 (w, x, y, z),表示旋转
    """
    # 根据轴顺序字符串,查表找到对应的 firstaxis、parity、repetition、frame
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 验证有效性
        firstaxis, parity, repetition, frame = axes

    # 计算对应的轴编号 (1-based),便于后续四元数分配
    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    # 根据 frame 设置,交换 ai 和 ak(第一和第三旋转角)
    if frame:
        ai, ak = ak, ai
    # 根据 parity 设置,aj 取负
    if parity:
        aj = -aj

    # 将角度减半,因为四元数旋转是基于半角的
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0

    # 预先计算正弦和余弦
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)

    # 辅助变量
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    # 创建空四元数数组
    q = np.empty((4,))
    # 如果中间轴重复(比如 zxz、xyx),使用这一套公式
    if repetition:
        q[0] = cj * (cc - ss)  # w
        q[i] = cj * (cs + sc)  # 第一个旋转轴分量
        q[j] = sj * (cc + ss)  # 第二个旋转轴分量
        q[k] = sj * (cs - sc)  # 第三个旋转轴分量
    else:
        # 中间轴不重复(比如 xyz、zyx)
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        # 根据 parity 决定是否取负
        q[j] *= -1.0
    return q


def quaternion_about_axis(angle, axis):
    """
    根据旋转角度和旋转轴,返回对应的四元数

    参数:
    angle : 旋转角度(单位:弧度)
    axis  : 长度为3的向量,表示旋转轴(应为单位向量)
    返回:q : 四元数 [w, x, y, z],表示绕给定轴旋转 angle 弧度的旋转
    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])  # 初始化四元数(虚部)
    qlen = vector_norm(q)  # 求旋转轴的长度(单位化之前)
    if qlen > _EPS:  # 若轴有效(非零)
        q *= math.sin(angle / 2.0) / qlen  # 虚部:sin(θ/2) * 归一化轴
    q[0] = math.cos(angle / 2.0)  # 实部:cos(θ/2)
    return q


def quaternion_matrix(quaternion):
    """
    将四元数转换为齐次旋转矩阵(4x4)

    参数:
    quaternion : 四元数 [w, x, y, z],应为单位四元数

    返回:
    旋转矩阵(4x4),右下角为 1,用于 3D 齐次坐标变换
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)  # 计算四元数模长平方

    if n < _EPS:
        return np.identity(4)  # 零四元数 => 单位矩阵

    q *= math.sqrt(2.0 / n)  # 单位化后放大 √2,方便构建矩阵
    q = np.outer(q, q)  # 外积,用于构造矩阵元素
    # 根据公式构造旋转矩阵
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """
    从旋转矩阵中提取四元数

    参数:
    matrix : 4x4 齐次旋转矩阵
    isprecise : 若为 True,表示输入矩阵是精确的正交旋转矩阵,将使用更快算法

    返回:
    quaternion : 四元数 [w, x, y, z]
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]  # 取前4x4子矩阵
    if isprecise:
        # 快速路径:使用矩阵迹(trace)提取
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            # 一般路径:通过特征值/特征向量提取
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # 构造对称矩阵 K(Hamilton 矩阵),其主特征向量是所需四元数
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # 计算最大特征值对应的特征向量
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]  # 注意顺序调整 [w, x, y, z]
    if q[0] < 0.0:
        np.negative(q, q)  # 保证实部为正,规范化
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """
    返回两个四元数的乘积

    参数:
    quaternion1 : 四元数1,形式为 [w, x, y, z]
    quaternion0 : 四元数0,形式为 [w, x, y, z]

    返回:
    结果四元数,形式为 [w, x, y, z]
    """
    w0, x0, y0, z0 = quaternion0  # 四元数0的分量
    w1, x1, y1, z1 = quaternion1  # 四元数1的分量
    # 使用四元数乘法公式计算结果四元数
    # 计算实部 w 计算虚部 x 计算虚部 y 计算虚部 z
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_conjugate(quaternion):
    """
    返回四元数的共轭
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)  # 创建四元数的副本
    np.negative(q[1:], q[1:])  # 将虚部 (x, y, z) 取负
    return q  # 返回共轭四元数


def quaternion_inverse(quaternion):
    """
    返回四元数的逆
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)  # 创建四元数的副本
    np.negative(q[1:], q[1:])  # 将虚部 (x, y, z) 取负
    return q / np.dot(q, q)  # 返回四元数的逆:共轭 / 模长的平方


def quaternion_real(quaternion):
    """
    返回四元数的实部
    四元数的实部就是四元数的第一个元素 w

    参数:
    quaternion : 四元数,形式为 [w, x, y, z]

    返回:
    实部 w,浮动类型
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """
    返回四元数的虚部
    四元数的虚部是四元数的后三个元素 [x, y, z]

    参数:
    quaternion : 四元数,形式为 [w, x, y, z]

    返回:
    虚部 [x, y, z],类型为 ndarray
    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    返回两个四元数之间的球面线性插值(SLERP)

    参数:
    quat0 : 起始四元数
    quat1 : 目标四元数
    fraction : 插值因子,范围 [0, 1]
    spin : 附加旋转圈数,默认 0
    shortestpath : 是否沿最短路径插值(默认是)

    返回:
    四元数 q,表示 quat0 到 quat1 的 fraction 位置的插值结果
    """
    q0 = _unit_vector(quat0[:4])  # 单位化起始四元数
    q1 = _unit_vector(quat1[:4])  # 单位化目标四元数
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)  # 计算两个单位四元数的点积(余弦夹角)
    if abs(abs(d) - 1.0) < _EPS:
        # 如果夹角接近 0° 或 180°,直接返回起始值(避免数值不稳定)
        return q0
    if shortestpath and d < 0.0:
        # 若启用最短路径且角度大于 90°,则取反方向
        d = -d
        np.negative(q1, q1)  # 反转目标四元数方向
    angle = math.acos(d) + spin * np.pi  # 插值总角度
    if abs(angle) < _EPS:
        return q0  # 若角度几乎为 0,直接返回
    isin = 1.0 / math.sin(angle)  # 预计算因子
    # 插值组合 q0 和 q1 的加权和
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1  # 合并为最终四元数
    return q0


def random_quaternion(rand=None):
    """
    返回一个单位四元数,其在四元数球面上均匀随机分布

    参数:
    rand : 可选,长度为 3 的数组或 None
           若为 None,则自动生成 3 个 [0, 1] 区间内的随机数
           否则使用传入的随机数
    返回:
    一个单位四元数 [w, x, y, z]
    """
    if rand is None:
        rand = np.random.rand(3)  # 生成三个随机数
    else:
        assert len(rand) == 3  # 确保输入长度为 3
    # 利用 Shoemake 算法进行均匀采样
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    # 返回四元数:[w, x, y, z]
    return np.array([np.cos(t2) * r2,
                     np.sin(t1) * r1,
                     np.cos(t1) * r1,
                     np.sin(t2) * r2])


def random_rotation_matrix(rand=None):
    """
    返回一个**均匀随机分布**的旋转矩阵(3D)

    参数:
    rand : array-like,可选
        可传入一个长度为3的数组,其中每个值在 [0, 1] 范围内
        用作生成四元数的随机数
        若为 None,则使用 np.random.rand 自动生成
    返回:
    一个 4x4 的齐次旋转矩阵(Homogeneous Rotation Matrix)
    """
    return quaternion_matrix(random_quaternion(rand))


class Arcball(object):
    """
    虚拟轨迹球控制类(Virtual Trackball Control)
    用于模拟在3D空间中的旋转,常用于图形学中的交互控制
    """

    def __init__(self, initial=None):
        """
        初始化虚拟轨迹球控制

        参数:
        initial : 四元数或旋转矩阵(可选)
            用于初始化轨迹球的旋转.如果为 None,初始化为单位四元数
        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = np.array([0.0, 0.0, 1.0])
        self._constrain = False
        if initial is None:
            self._qdown = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            initial = np.array(initial, dtype=np.float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4,):
                initial /= vector_norm(initial)
                self._qdown = initial
            else:
                raise ValueError("initial not a quaternion or matrix")
        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """
        放置轨迹球,当窗口大小发生变化时使用

        参数:
        center : 序列[2]
            轨迹球中心的窗口坐标.
        radius : float
            轨迹球的半径,单位为窗口坐标.
        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """
        设置旋转约束的轴

        参数:
        axes : 可变参数
            约束旋转的轴,传入一系列的向量.如果为空,表示没有约束
        """
        if axes is None:
            self._axes = None
        else:
            self._axes = [_unit_vector(axis) for axis in axes]

    @property
    def constrain(self):
        """
        返回当前是否启用了轴约束模式
        """
        return self._constrain

    @constrain.setter
    def constrain(self, value):
        """
        设置轴约束模式状态
        """
        self._constrain = bool(value)

    def down(self, point):
        """
        设置初始的鼠标窗口坐标,并选择是否约束轴

        参数:
        point : 当前鼠标点击位置的窗口坐标
        """
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)  # 将点击点映射到球面
        self._qdown = self._qpre = self._qnow  # 记录初始状态
        if self._constrain and self._axes is not None:  # 如果启用了约束轴
            self._axis = arcball_nearest_axis(self._vdown, self._axes)  # 找到与点击点最接近的旋转轴
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)  # 将点击点约束到该轴
        else:
            self._axis = None

    def drag(self, point):
        """
        更新当前的鼠标窗口坐标,并计算旋转

        参数:
        point : 当前鼠标拖动位置的窗口坐标
        """
        vnow = arcball_map_to_sphere(point, self._center, self._radius)  # 将拖动点映射到球面
        if self._axis is not None:  # 如果启用了轴约束
            vnow = arcball_constrain_to_axis(vnow, self._axis)  # 将拖动点约束到该轴
        self._qpre = self._qnow  # 记录上一个状态
        t = np.cross(self._vdown, vnow)  # 计算两点的叉积
        if np.dot(t, t) < _EPS:  # 如果旋转角度很小,保持当前状态
            self._qnow = self._qdown
        else:
            q = [np.dot(self._vdown, vnow), t[0], t[1], t[2]]  # 计算旋转四元数
            self._qnow = quaternion_multiply(q, self._qdown)  # 更新当前四元数

    def next(self, acceleration=0.0):
        """
        按照上次拖动的方向继续旋转

        参数:
        acceleration : float,可选
            旋转的加速度,控制旋转的速度
        """
        q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)  # 使用球面线性插值
        self._qpre, self._qnow = self._qnow, q  # 更新四元数状态

    def matrix(self):
        """
        返回当前的齐次旋转矩阵(4x4)

        返回:
        旋转矩阵(4x4)
        """
        return quaternion_matrix(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """
    从窗口坐标返回单位球坐标

    参数:
    point : 序列 [2]
        鼠标点击或拖动的窗口坐标
    center : 序列 [2]
        轨迹球的中心位置,窗口坐标系中的坐标
    radius : float
        轨迹球的半径

    返回:
    numpy 数组 [3]
        对应的单位球坐标
    """
    # 计算鼠标位置与中心的相对位置,并归一化
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0 * v0 + v1 * v1
    if n > 1.0:
        # 如果鼠标位置超出了球面,则将其映射到球面上
        n = math.sqrt(n)
        return np.array([v0 / n, v1 / n, 0.0])
    else:
        # 如果鼠标位置在球面内部,返回单位球坐标
        return np.array([v0, v1, math.sqrt(1.0 - n)])


def arcball_constrain_to_axis(point, axis):
    """
    返回与给定轴垂直的球面点

    参数:
    point : 序列 [3]
        要约束的点,单位球上的一个点
    axis : 序列 [3]
        约束的轴,通常是一个单位向量

    返回:
    numpy 数组 [3]
        约束后的单位球坐标点,垂直于给定轴
    """
    # 创建副本避免修改原数据
    v = np.array(point, dtype=np.float64, copy=True)
    a = np.array(axis, dtype=np.float64, copy=True)

    # 将点投影到与给定轴垂直的平面上
    v -= a * np.dot(a, v)
    n = vector_norm(v)  # 计算该向量的模
    # 如果向量不为零,进行归一化处理
    if n > _EPS:
        if v[2] < 0.0:  # 如果垂直方向小于零,则反转
            np.negative(v, v)
        v /= n  # 归一化
        return v
    # 如果轴垂直于Z轴(即a[2] == 1),返回与X轴平行的单位向量
    if a[2] == 1.0:
        return np.array([1.0, 0.0, 0.0])
    # 否则返回垂直于给定轴的单位向量
    return _unit_vector([-a[1], a[0], 0.0])


def arcball_nearest_axis(point, axes):
    """
    返回与点最接近的轴

    参数:
    point : 序列 [3]
        3D 空间中的一个点
    axes : 可迭代对象 [n, 3]
        一组单位向量,表示不同的轴

    返回:
    numpy 数组 [3]
        最接近给定点的单位轴
    """
    point = np.array(point, dtype=np.float64, copy=False)  # 将点转换为 numpy 数组
    nearest = None  # 初始化最近的轴
    mx = -1.0  # 初始化最大值为负无穷
    for axis in axes:
        # 将点约束到轴上,计算它与原点的点积
        t = np.dot(arcball_constrain_to_axis(point, axis), point)
        # 如果当前点积大于最大值,则更新最近的轴
        if t > mx:
            nearest = axis
            mx = t
    return nearest


def vector_norm(data, axis=None, out=None):
    """
    返回给定 ndarray 沿指定轴的欧几里得范数(即向量的长度)

    参数:
    log_filename : ndarray
        输入数据,可以是多维数组
    axis : int, 可选
        要计算范数的轴.如果为 None,则计算整个数组的范数
    out : ndarray, 可选
        输出数组,用于存储计算结果

    返回:
    float 或 ndarray
        输入数组的欧几里得范数
    """
    data = np.array(data, dtype=np.float64, copy=True)  # 将数据转换为 numpy 数组
    # 如果没有指定输出数组
    if out is None:
        if data.ndim == 1:  # 如果是 1D 数组,直接计算范数
            return math.sqrt(np.dot(data, data))
        data *= data  # 数据平方
        out = np.atleast_1d(np.sum(data, axis=axis))  # 沿指定轴计算平方和
        np.sqrt(out, out)  # 取平方根
        return out  # 返回范数
    else:
        data *= data  # 数据平方
        np.sum(data, axis=axis, out=out)  # 沿指定轴计算平方和
        np.sqrt(out, out)  # 取平方根


def _unit_vector(data, axis=None, out=None):
    """
    返回沿指定轴归一化的 ndarray,得到单位向量

    参数:
    log_filename : ndarray
        输入数据,可以是多维数组
    axis : int, 可选
        沿哪个轴进行归一化.如果为 None,则返回整个数据的单位向量
    out : ndarray, 可选
        输出数组,用于存储归一化后的结果

    返回:
    ndarray
        归一化后的单位向量
    """
    # 如果没有指定输出数组
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)  # 将数据转换为 numpy 数组
        # 如果是 1D 数组,直接归一化
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        # 如果指定了输出数组
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out  # 将数据赋值给输出数组
    length = np.atleast_1d(np.sum(data * data, axis))  # 计算向量的模长(平方和)
    np.sqrt(length, length)  # 取平方根
    if axis is not None:  # 如果指定了轴
        length = np.expand_dims(length, axis)  # 扩展维度,使其与数据维度匹配
    data /= length  # 将数据除以模长,得到单位向量
    # 将数据除以模长,得到单位向量
    if out is None:
        return data


def gen_icorotmats(icolevel=1,
                   rotation_interval=math.radians(45),
                   crop_normal=np.array([0, 0, 1]),
                   crop_angle=math.pi,
                   toggleflat=False):
    """
    使用 icosphere 和旋转角度生成旋转矩阵,每个原点-顶点向量进行旋转

    :param icolevel: icosphere的层级,默认值1,表示42个顶点
    :param rotation_interval: 每个旋转的间隔角度,默认为45度
    :param crop_normal: 用于裁剪的法向量,默认值是[0, 0, 1]
    :param crop_angle: 裁剪的角度(裁剪一个圆锥部分),默认值是π(180度)
    :param toggleflat: 是否将结果展平为一维数组,默认值为False

    :return: 一个包含旋转矩阵的列表,每个列表包含多个旋转矩阵,旋转矩阵的数量取决于旋转间隔角度

    作者: weiwei
    日期: 20191015osaka
    """
    returnlist = []  # 用于存储旋转矩阵的结果
    icos = trm_creation.icosphere(icolevel)  # 创建一个 icosphere
    # 遍历 icosphere 的每个顶点
    for vert in icos.vertices:
        # 如果裁剪角度小于 180度,检查点与法线的夹角是否大于裁剪角度,如果是,则跳过该点
        if crop_angle < math.pi:
            if angle_between_vectors(vert, crop_normal) > crop_angle:
                continue
        # 计算与顶点相反的方向的向量,作为旋转的基准
        z = -vert
        # 计算与 z 向量正交的向量
        x = orthogonal_vector(z)
        # 计算 y 向量,它与 z 和 x 向量正交,并且是单位向量
        y = unit_vector(np.cross(z, x))
        # 创建一个 3x3 的单位矩阵,然后将 x, y, z 向量作为矩阵的列
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        returnlist.append([])  # 为当前顶点创建一个新的列表来存储旋转矩阵
        # 生成多个旋转矩阵,通过在 z 轴上旋转不同的角度
        for angle in np.linspace(0, 2 * math.pi, int(2 * math.pi / rotation_interval), endpoint=False):
            # 计算旋转矩阵并将其添加到返回列表中
            returnlist[-1].append(np.dot(rotmat_from_axangle(z, angle), temprotmat))
    # 如果 toggleflat 为 True,将返回列表展平成一个一维数组
    if toggleflat:
        return functools.reduce(operator.iconcat, returnlist, [])
    # 返回包含所有旋转矩阵的列表
    return returnlist


if __name__ == '__main__':
    # start_pos = np.array([1, 0, 0])
    # start_rotmat = np.eye(3)
    # goal_pos = np.array([2, 0, 0])
    # goal_rotmat = np.eye(3)
    # pos_list, rotmat_list = interplate_pos_rotmat(start_pos, start_rotmat, goal_pos, goal_rotmat, granularity=3)
    # print(pos_list, rotmat_list)

    import math
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rots_candidate = np.array(gen_icorotmats(icolevel=3,
                                             rotation_interval=np.radians(360),
                                             crop_normal=-np.array([0, 0, 1]),
                                             crop_angle=np.radians(15),
                                             toggleflat=True))
    # rots_candidate[..., [0, 1]] = rots_candidate[..., [1, 2]]

    for rot in rots_candidate:
        gm.gen_frame(rotmat=rotmat_from_axangle(np.array([0, 0, 1]), np.radians(180)).dot(rot)).attach_to(base)
    base.run()
