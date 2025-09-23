import numpy as np
import math
import itertools


def centerPoftrangle(a, b, c):
    '''
    计算三角形的中心点,并返回包含三个顶点和中心点的数组

    :param a: 三角形的第一个顶点
    :param b: 三角形的第二个顶点
    :param c: 三角形的第三个顶点
    :return: 包含三个顶点和中心点的数组
    '''
    x_middle = (a[0] + b[0] + c[0]) / 3
    y_middle = (a[1] + b[1] + c[1]) / 3
    z_middle = (a[2] + b[2] + c[2]) / 3
    d = np.array([x_middle, y_middle, z_middle])
    surface = np.concatenate((a, b, c, d), axis=0)
    return surface


def centerPoint(a):
    '''
    计算给定顶点列表的中心点

    :param a: 顶点列表
    :return: 中心点的坐标
    '''
    vertices = []
    for a in a:
        vertices.append(a)
    verticesnp = np.array(vertices)
    verticesnpsum = np.sum(verticesnp, axis=0) / len(vertices)
    # print(verticesnpsum)
    d = np.array([verticesnpsum[0], verticesnpsum[1], verticesnpsum[2]])
    return d


def centerPointwithArea(a, area):
    '''
    计算加权面积的中心点

    :param a: 顶点列表
    :param area: 每个顶点的权重面积
    :return: 加权中心点的坐标
    '''
    vertices = []
    for i, a in enumerate(a):
        vertices.append(a * area[i])
    verticesnp = np.array(vertices)
    verticesnpsum = np.sum(verticesnp, axis=0) / sum(area)
    # print(verticesnpsum)
    d = np.array([verticesnpsum[0], verticesnpsum[1], verticesnpsum[2]])
    return d


def creatlist(n, m):
    '''
    创建一个 n 行 m 列的零矩阵

    :param n: 行数
    :param m: 列数
    :return: n 行 m 列的零矩阵
    '''
    n = n
    m = m
    matrix = [None] * n
    for i in range(len(matrix)):
        matrix[i] = [0] * m
    return matrix


def getrighthanded(coordinate, holdingpair):
    '''
    根据给定的坐标和持有对,计算右手坐标系

    :param coordinate: 坐标列表
    :param holdingpair: 持有对列表
    :return: 右手坐标系和面列表
    '''
    coordinate = coordinate
    faces = []
    x_axis0 = coordinate[0]
    x_tempcheck = np.cross(coordinate[1], coordinate[2])
    x_check = np.dot(x_axis0, x_tempcheck)
    if x_check > 0:
        y_axis0 = coordinate[1]
        z_axis0 = coordinate[2]
        # faces.append([holdingpair[0],holdingpair[1],holdingpair[2]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis0 = coordinate[2]
        z_axis0 = coordinate[1]
        # faces.append([holdingpair[0], holdingpair[2], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate0 = np.array([x_axis0, y_axis0, z_axis0])

    x_axis1 = coordinate[1]
    x_tempcheck = np.cross(coordinate[0], coordinate[2])
    x_check = np.dot(x_axis1, x_tempcheck)
    if x_check > 0:
        y_axis1 = coordinate[0]
        z_axis1 = coordinate[2]
        # faces.append([holdingpair[1], holdingpair[0], holdingpair[2]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis1 = coordinate[2]
        z_axis1 = coordinate[0]
        # faces.append([holdingpair[0], holdingpair[2], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate1 = np.array([x_axis1, y_axis1, z_axis1])

    x_axis2 = coordinate[2]
    x_tempcheck = np.cross(coordinate[0], coordinate[1])
    x_check = np.dot(x_axis2, x_tempcheck)
    if x_check > 0:
        y_axis2 = coordinate[0]
        z_axis2 = coordinate[1]
        # faces.append([holdingpair[2], holdingpair[0], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis2 = coordinate[1]
        z_axis2 = coordinate[0]
        # faces.append([holdingpair[2], holdingpair[1], holdingpair[0]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate2 = np.array([x_axis2, y_axis2, z_axis2])
    righthandedco = []
    righthandedco.append(coordinate0.T)
    righthandedco.append(coordinate1.T)
    righthandedco.append(coordinate2.T)
    # print("righthandedco",righthandedco)
    # faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    return righthandedco, faces


def listnorepeat(list):
    '''
    移除列表中的重复项,同时保持原始顺序

    :param lst: 输入列表
    :return: 不含重复项的列表
    '''
    list = list
    listnorepeat = []
    for i, index in enumerate(list):
        # print("listnorepeat", listnorepeat)
        # print("index", index)
        ad = []
        if i == 0:
            listnorepeat.append(index)
        else:
            for j in range(len(listnorepeat)):
                # if j!=i:
                if (index == listnorepeat[j]).all():
                    ad.append(False)
                else:
                    ad.append(True)
            if all(ad):
                listnorepeat.append(index)
    return listnorepeat


def distance(a, b):
    '''
    计算两个点之间的距离

    :param a: 第一个点的坐标
    :param b: 第二个点的坐标
    :return: 两点之间的距离
    '''
    if len(a) == 2:
        distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
        return distance
    if len(a) == 3:
        distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))
        return distance


def TSP(obj):
    '''
    解决旅行商问题,找到最短路径

    :param obj: 点的列表
    :return: 最短路径的点序列
    '''
    number = len(obj)
    seq_number = np.arange(0, number, 1).tolist()
    iter = itertools.permutations(seq_number, number)
    sequence = list(iter)
    temdistance = []
    for i in range(len(sequence)):
        temdist = 0
        for j in range(len(sequence[i])):
            if j < len(sequence[i]) - 1:
                temdist += distance(obj[sequence[i][j]], obj[sequence[i][j + 1]])
            if j == len(sequence[i]) - 1:
                temdist += distance(obj[sequence[i][j]], obj[sequence[i][0]])
        temdistance.append(temdist)
    mindistance = temdistance.index(min(temdistance))
    minsquence = list(sequence[mindistance])
    outputobj = np.copy(obj)
    for i in range(outputobj.shape[0]):
        outputobj[i] = obj[minsquence[i]]
    # outputobj = outputobj.tolist()
    # print("outputobj",outputobj,type(outputobj))
    # print(minsquence,type(minsquence))
    return outputobj


def rowtocolume(array):
    '''
    将数组的行转换为列

    :param array: 输入数组
    :return: 转换后的列数组
    '''
    n = array.shape[0]
    new = np.zeros((n, 1))
    for i in range(n):
        new[i][0] = array[i]
    return new


def removeduplication(array, decimals):
    '''
    移除数组中的重复项

    :param array: 输入数组
    :param decimals: 保留的小数位数
    :return: 不含重复项的数组
    '''
    b = np.round(array, decimals=decimals).tolist()[:-1]
    c = []
    for i in b:
        if i not in c:
            c.append(i)
    return c


def getsurfaceequation(normal, pnt):
    '''
    使用已知法线和点生成平面方程

    :param normal: 法线向量
    :param pnt: 点坐标
    :return: 平面方程的系数向量 [A, B, C, D] Ax+By+Cz+D=0
    A = surfaceequation[0];B = surfaceequation[1]L;C = surfaceequation[2];D = surfaceequation[3]
    '''
    pnt = pnt
    normal = normal
    D = -np.dot(normal, pnt)
    surfaceequation = [normal[0], normal[1], normal[2], D]
    return surfaceequation


def getpointfromonesurface(surfaceequation, x, y):
    '''
    使用给定的 X, Y 和平面方程获取点的坐标

    :param surfaceequation: 平面方程的系数向量
    :param x: X 坐标
    :param y: Y 坐标
    :return: 点的坐标 [x, y, z]
    '''
    import sympy
    z = sympy.Symbol('z')
    # x = sympy.Symbol('x')
    # y = sympy.Symbol('y')
    x = x
    y = y
    expr1 = x * surfaceequation[0] + y * surfaceequation[1] + z * surfaceequation[2] + surfaceequation[3]
    result = sympy.solve(expr1, z)[0]
    return [x, y, result]


if __name__ == "__main__":
    # a = [[0,0,0], [100,100,0], [100,0,0], [0,100,0]]
    #     #
    #     # centerPoint(a)
    #     a = np.array([[1,0,0],[0,0,1],[0,1,0]])
    #     a = np.array([[0.0, 3], [.0, 2.0], [2.0, 2.0],[1.0,0.0],[1.0,1.0],[2.0,1.0]])
    a = np.array([[30., 42.42640615], [8.78679693, 42.42640615], [15., 21.21320307], [-6.21320307, 21.21320307]])
    # a = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
    TSP(a)
    # print(d)
    # isitrighthanded(a)
