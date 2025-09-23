import time
import numpy as np
import math
import matplotlib.pyplot as plot


def concentric_circle_hex_polar(layer, radians, start_rot_angle=0.):
    """
    生成同心圆的六边形极坐标顶点

    :param layer: 层数
    :param radians: 半径
    :param start_rot_angle: 起始旋转角度
    :return: x 和 y 坐标列表
    """

    def get_param(layer_id):
        """
        计算给定层的角度和半径参数

        :param layer_id: 当前层的索引
        :return: 角度列表和半径列表
        """
        radians_base = 0.866025 * radians * (layer_id + 1)  # 基础半径,0.866025 是 sqrt(3)/2 的近似值
        n_list = np.linspace(layer_id - 1, 0, int((layer_id + 1) / 2))  # 生成从 layer_id-1 到 0 的线性间隔
        angle_diff = np.append(np.array([math.pi / 6]), np.arctan(n_list / ((layer_id + 1) * 1.732051)))

        # 计算角度差,1.732051 是 sqrt(3) 的近似值
        # print("angle_diff:", angle_diff)
        angle_minus = np.zeros(len(angle_diff))
        angle_minus[:-1] = angle_diff[1:]
        angle_half = angle_diff - angle_minus
        angle_list = np.append(angle_half, np.zeros(int(layer_id / 2)))
        angle_list = angle_list + angle_list[::-1]
        # print("angle_list:", len(angle_list), angle_list)
        angle_diff_total = np.append(angle_diff[1:], angle_diff[1:][::-1][(layer_id % 2):])
        radians_list = np.append(radians_base / np.cos(angle_diff_total), radians * (layer_id + 1))
        # print("radiasn_list:", len(radians_list), radians_list)
        return angle_list, radians_list

    def get_pose_from_angle(angle_list, radians_list):
        """
        根据角度和半径计算顶点的 x 和 y 坐标

        :param angle_list: 角度列表
        :param radians_list: 半径列表
        :return: x 和 y 坐标列表
        """

        # angle_list[0]  += start_rot_angle
        angle_list_total = np.cumsum(np.tile(angle_list, 6))  # 累积求和,重复 6 次以形成完整的六边形
        angle_list_total = angle_list_total + np.array([start_rot_angle]).repeat(len(angle_list_total))
        radians_list_total = np.tile(radians_list, 6)  # 重复 6 次以形成完整的六边形
        x_list = radians_list_total * np.sin(angle_list_total)  # 计算 x 坐标
        y_list = radians_list_total * np.cos(angle_list_total)  # 计算 y 坐标
        return x_list, y_list

    x_list = np.array([])
    y_list = np.array([])
    for layer_id in range(layer):
        # print("layer_id", layer_id)
        # 对每一层计算角度和半径
        angle_list, radians_list = get_param(layer_id)
        # 根据角度和半径计算顶点坐标
        x_layer, y_layer = get_pose_from_angle(angle_list, radians_list)
        # 将当前层的顶点坐标添加到总列表中
        x_list = np.append(x_list, x_layer)
        y_list = np.append(y_list, y_layer)

    return x_list, y_list


def concentric_circle_hex_equipartition(layer, radians, start_rot_angle=0.):
    """
    生成同心圆的六边形等分顶点

    :param layer: 层数
    :param radians: 半径
    :param start_rot_angle: 起始旋转角度
    :return: x 和 y 坐标列表
    """

    def get_hex(layer_id):
        angle_list = np.arange(start_rot_angle, (math.pi * 2 + start_rot_angle), math.pi / 3)
        angle_list = np.append(angle_list, start_rot_angle)
        x_vertex = np.sin(angle_list) * radians * (layer_id + 1)
        y_vertex = np.cos(angle_list) * radians * (layer_id + 1)
        return x_vertex, y_vertex

    x_list = np.array([])
    y_list = np.array([])
    for layer_id in range(layer):
        x_vertex, y_vertex = get_hex(layer_id)
        for i in range(6):
            x_list = np.append(x_list, np.linspace(x_vertex[i], x_vertex[i + 1], num=layer_id + 1, endpoint=False))
            y_list = np.append(y_list, np.linspace(y_vertex[i], y_vertex[i + 1], num=layer_id + 1, endpoint=False))
    return x_list, y_list


def gen_regpoly(radius, nedges=12):
    """
    生成规则多边形的顶点

    :param radius: 半径
    :param nedges: 边数
    :return: 顶点的 x 和 y 坐标
    """
    angle_list = np.linspace(0, np.pi * 2, nedges + 1, endpoint=True)
    x_vertex = np.sin(angle_list) * radius
    y_vertex = np.cos(angle_list) * radius
    return np.column_stack((x_vertex, y_vertex))


def gen_2d_isosceles_verts(nlevel, edge_length, nedges=12):
    """
    生成二维等腰多边形的顶点

    :param nlevel: 层数
    :param edge_length: 边长
    :param nedges: 边数
    :return: 顶点的坐标数组
    """
    xy_array = np.asarray([[0, 0]])
    for level in range(nlevel):
        xy_vertex = gen_regpoly(radius=edge_length * (level + 1), nedges=nedges)
        for i in range(nedges):
            xy_array = np.append(xy_array,
                                 np.linspace(xy_vertex[i, :], xy_vertex[i + 1, :], num=level + 1, endpoint=False),
                                 axis=0)
    return xy_array


def gen_2d_equilateral_verts(nlevel, edge_length):
    """
    生成二维等边多边形的顶点

    :param nlevel: 层数
    :param edge_length: 边长
    :return: 顶点的坐标数组
    """
    return gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=6)


def gen_3d_isosceles_verts(pos, rotmat, nlevel=5, edge_length=0.001, nedges=12):
    """
    生成三维等腰多边形的顶点

    :param pos: 位置
    :param rotmat: 旋转矩阵
    :param nlevel: 层数
    :param edge_length: 边长
    :param nedges: 边数
    :return: 顶点的坐标数组
    """
    xy_array = gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=nedges)
    xyz_array = np.pad(xy_array, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    return rotmat.dot((xyz_array + pos).T).T


def gen_3d_equilateral_verts(pos, rotmat, nlevel=5, edge_length=0.001):
    """
    生成三维等边多边形的顶点

    :param pos: 位置
    :param rotmat: 旋转矩阵
    :param nlevel: 层数
    :param edge_length: 边长
    :return: 顶点的坐标数组
    """
    return gen_3d_isosceles_verts(pos=pos, rotmat=rotmat, nlevel=nlevel, edge_length=edge_length, nedges=6)


def gen_2d_equilaterial_verts(nlevel, edge_length):
    """
    生成二维等边多边形的顶点

    :param nlevel: 层数
    :param edge_length: 边长
    :return: 顶点的坐标数组
    """
    nangle = 12
    levels = np.arange(1, nlevel + 1, 1) * edge_length
    angles = np.linspace(0, np.pi * 2, nangle + 1, endpoint=True)
    x_verts = np.outer(levels, np.sin(angles)).flatten()
    y_verts = np.outer(levels, np.cos(angles)).flatten()
    xy_vertex = np.row_stack((x_verts, y_verts)).T
    xy_list = np.empty((0, 2))
    for level in range(nlevel):
        for i in range(nangle):
            xy_list = np.append(xy_list,
                                np.linspace(xy_vertex[level * (nangle + 1) + i, :],
                                            xy_vertex[level * (nangle + 1) + i + 1, :], num=level + 1, endpoint=False),
                                axis=0)
    return xy_list


if __name__ == "__main__":
    tic = time.time()
    for i in range(200):
        x_list, y_list = concentric_circle_hex_polar(5, 1, math.pi / 8)
    toc1 = time.time()
    print(toc1 - tic)
    tic = time.time()
    for i in range(200):
        x_list, y_list = concentric_circle_hex_equipartition(5, 1, math.pi / 8)
    toc1 = time.time()
    print(toc1 - tic)
    tic = time.time()
    for i in range(200):
        xy_list = gen_2d_isosceles_verts(5, 1, 12)
    toc1 = time.time()
    print(toc1 - tic)
    # for i in range(200):
    #     xy_list = gen_2d_equilaterial_verts(5, 1)
    # toc1 = time.time()
    # print(toc1 - tic)

    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')

    plot.plot(xy_list[:, 0], xy_list[:, 1], "o-")
    # plot.plot(x_list[:], y_list[:], "o-")
    plot.show()
