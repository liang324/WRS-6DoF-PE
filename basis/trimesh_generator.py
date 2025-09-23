# 生成一组三角网格对象,以米弧度为单位

import math
import numpy as np
import basis.trimesh.primitives as tp
import basis.trimesh as trm
import basis.robot_math as rm
import shapely.geometry as shpg


def gen_box(extent=np.array([1, 1, 1]), homomat=np.eye(4)):
    """
    生成一个立方体(盒子)对象
    
    :param extent: 盒子的尺寸,x、y、z(原点为 0)
    :param homomat: 旋转和平移的变换矩阵
    :return: 一个 Trimesh 对象(原始几何体)
    
    作者: weiwei
    日期: 20191228大阪
    """
    return tp.Box(box_extents=extent, box_transform=homomat)


def gen_cylinder(radius=0.01, height=0.1, section=100, homomat=np.eye(4)):
    """
    生成一个圆柱体的 Trimesh 对象

    :param radius: 圆柱的半径
    :param height: 圆柱的高度
    :param sections: 圆柱的分段数
    :param homomat: 齐次变换矩阵,用于旋转和位移
    :return: 一个 Trimesh 对象(圆柱体)

    author: hu
    date: 20220113osaka
    """
    return tp.Cylinder(radius=radius, height=height, sections=section, homomat=homomat)


def gen_stick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, type="rect", sections=8):
    """
    生成矩形或圆形的棍棒,内部调用 `genrectstick` 或 `genroundstick`

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒的厚度,默认为 0.005 米
    :param type: 棍棒的类型,可以是 "rect"(矩形)或 "round"(圆形)
    :param sections: 用来近似圆柱体的离散化部分数
    :return: 返回相应类型的棍棒几何体

    作者: weiwei
    日期: 20191228大阪
    """
    if type == "rect":
        return gen_rectstick(spos, epos, thickness, sections=sections)
    if type == "round":
        return gen_roundstick(spos, epos, thickness, count=[sections / 2.0, sections / 2.0])


def gen_rectstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=.005, sections=8):
    """
    生成矩形的棍棒(圆柱体)

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒的厚度,默认为 0.005 米
    :param sections: 用来近似圆柱体的离散化部分数
    :return: 一个 Trimesh 对象(原始几何体)

    作者: weiwei
    日期: 20191228大阪
    """
    pos = spos
    height = np.linalg.norm(epos - spos)
    if np.allclose(height, 0):
        rotmat = np.eye(3)
    else:
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    return tp.Cylinder(height=height, radius=thickness / 2.0, sections=sections, homomat=homomat)


def gen_roundstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, count=[8, 8]):
    """
    生成一个圆形棍棒(胶囊形状)

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒的厚度,默认为 0.005 米
    :param count: 用来近似圆柱体的离散化部分数,默认为 [8, 8]
    :return: 一个 Trimesh 对象(原始几何体,胶囊形状)

    作者: weiwei
    日期: 20191228大阪
    """
    pos = spos
    height = np.linalg.norm(epos - spos)  # 计算棍棒的高度
    if np.allclose(height, 0):
        rotmat = np.eye(3)  # 如果起始点和结束点相同,使用单位矩阵
    else:
        # 计算从 [0, 0, 1] 到目标方向的旋转矩阵
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    # 获取旋转和平移的齐次变换矩阵
    homomat = rm.homomat_from_posrot(pos, rotmat)
    # 创建一个胶囊体(圆形棍棒)对象,并返回
    return tp.Capsule(height=height, radius=thickness / 2.0, count=count, homomat=homomat)

def gen_capsule(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=0.005, count=[8, 8]):
    """
    生成一个胶囊体的 Trimesh 对象

    :param spos: 起始位置
    :param epos: 结束位置
    :param radius: 胶囊的半径
    :param count: 胶囊的分段数
    :return: 一个 Trimesh 对象(胶囊体)

    author: weiwei
    date: 20191228osaka
    """
    pos = spos
    height = np.linalg.norm(epos - spos)
    # 计算旋转矩阵
    if np.allclose(height, 0):
        rotmat = np.eye(3)
    else:
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    # 生成齐次变换矩阵
    homomat = rm.homomat_from_posrot(pos, rotmat)
    return tp.Capsule(height=height, radius=radius, count=count, homomat=homomat)


def gen_section(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), height_vec=np.array([0, 0, 1]), height=0.01,
                angle=30, section=8):
    """
    生成一个截面的 Trimesh 对象

    :param spos: 起始位置
    :param epos: 结束位置
    :param height_vec: 高度方向向量
    :param height: 截面的高度
    :param angle: 截面的角度
    :param section: 截面的分段数
    :return: 一个 Trimesh 对象(截面)

    author: hu
    date: 20240617
    """
    pos = spos
    direction = rm.unit_vector(epos - spos)
    length = np.linalg.norm(epos - spos)
    height = height

    # 计算旋转矩阵
    if np.allclose(height, 0):
        rotmat_goal = np.eye(3)
    else:
        rotmat_goal = rm.rotmat_from_two_axis(direction, rm.unit_vector(height_vec), "xz")

    rotmat = rotmat_goal
    homomat = rm.homomat_from_posrot(pos, rotmat)

    # 计算中心偏移
    center_offset = - (rotmat[:, 2] * height / 2)
    center_offset_homo = rm.homomat_from_posrot(center_offset, np.eye(3))
    homomat = center_offset_homo.dot(homomat)
    # 计算边界方向
    angle_rad = np.deg2rad(angle)
    direction_boundary = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -angle_rad / 2), [1, 0, 0])
    # 生成曲线点
    curve_pnts = [(np.array([0, 0, 0]) + np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), i * angle_rad / section),
                                                direction_boundary) * length)[:2] for i in range(section + 1)]
    curve_pnts.append(np.array([0, 0]))  # 闭合曲线
    # 创建多边形并进行挤出
    extrude_polygon = shpg.Polygon(curve_pnts)
    extrude_transform = homomat
    extrude_height = height
    return tp.Extrusion(extrude_polygon=extrude_polygon, extrude_transform=extrude_transform,
                        extrude_height=extrude_height)


def gen_dashstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, lsolid=None, lspace=None,
                  sections=8, sticktype="rect"):
    """
    生成带有实心部分和空心部分的棍棒(具有间隔的矩形或圆形棍棒)

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒的厚度,默认为 0.005 米
    :param lsolid: 实心部分的长度,如果为 None,则默认为 `thickness * solidweight`
    :param lspace: 空心部分的长度,如果为 None,则默认为 `thickness * spaceweight`
    :param sections: 用来近似圆柱体的离散化部分数
    :param sticktype: 棍棒的类型,默认为 "rect"(矩形)
    :return: 由多个分段棍棒组成的 Trimesh 对象

    作者: weiwei
    日期: 20191228大阪
    """
    solidweight = 1.6  # 实心部分的权重因子
    spaceweight = 1.07  # 空心部分的权重因子
    if not lsolid:
        lsolid = thickness * solidweight  # 设置实心部分的长度
    if not lspace:
        lspace = thickness * spaceweight  # 设置空心部分的长度
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)  # 计算从起始点到结束点的方向和长度
    nstick = math.floor(length / (lsolid + lspace))  # 计算可以分成多少段
    vertices = np.empty((0, 3))  # 用于存储顶点的数组
    faces = np.empty((0, 3))  # 用于存储面的数组
    for i in range(0, nstick):
        # 计算每段棍棒的起始位置
        tmp_spos = spos + (lsolid * direction + lspace * direction) * i
        tmp_stick = gen_stick(spos=tmp_spos,
                              epos=tmp_spos + lsolid * direction,
                              thickness=thickness,
                              type=sticktype,
                              sections=sections)
        tmp_stick_faces = tmp_stick.faces + len(vertices)  # 更新面的索引
        vertices = np.vstack((vertices, tmp_stick.vertices))  # 合并顶点
        faces = np.vstack((faces, tmp_stick_faces))  # 合并面
    # 处理最后一段棍棒
    tmp_spos = spos + (lsolid * direction + lspace * direction) * nstick
    tmp_epos = tmp_spos + lsolid * direction
    final_length, _ = rm.unit_vector(tmp_epos - spos, toggle_length=True)
    if final_length > length:
        tmp_epos = epos  # 如果最后一段棍棒的长度超过预期,调整结束位置
    tmp_stick = gen_stick(spos=tmp_spos,
                          epos=tmp_epos,
                          thickness=thickness,
                          type=sticktype,
                          sections=sections)
    tmp_stick_faces = tmp_stick.faces + len(vertices)  # 更新最后一段面的索引
    vertices = np.vstack((vertices, tmp_stick.vertices))  # 合并最后一段的顶点
    faces = np.vstack((faces, tmp_stick_faces))  # 合并最后一段的面
    return trm.Trimesh(vertices=vertices, faces=faces)  # 返回最终的 Trimesh 对象


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.02, subdivisions=2):
    """
    生成一个球体对象

    :param pos: 球体的中心坐标,1x3 的 numpy 数组
    :param radius: 球体的半径,默认为 0.02 米
    :param subdivisions: 球体的离散化层级,用于 icosphere 的细分
    :return: 一个 Trimesh 对象(球体)

    作者: weiwei
    日期: 20191228大阪
    """
    return tp.Sphere(sphere_radius=radius, sphere_center=pos, subdivisions=subdivisions)


def gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), subdivisions=5):
    """
    生成一个椭球体对象

    :param pos: 椭球体的中心坐标,1x3 的 numpy 数组
    :param axmat: 3x3 矩阵,矩阵的每一列代表椭球体的一个轴
    :param subdivisions: icosphere 的细分层级,用于生成椭球体的离散化
    :return: 一个 Trimesh 对象(椭球体)

    作者: weiwei
    日期: 20191228大阪
    """
    sphere = tp.Sphere(sphere_radius=1, sphere_center=np.zeros(3), subdivisions=subdivisions)  # 创建单位球体
    vertices = axmat.dot(sphere.vertices.T).T  # 根据椭球的轴矩阵对球体顶点进行变换
    vertices = vertices + pos  # 平移球体到指定位置
    return trm.Trimesh(vertices=vertices, faces=sphere.faces)  # 返回椭球体的 Trimesh 对象


def gen_dumbbell(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, subdivisions=1):
    """
    生成一个哑铃形状(由棍棒和两个球组成)

    注意:返回棍棒+起始球+结束球也可以,但会稍微慢一些
    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒的厚度,默认为 0.005 米
    :param sections: 棍棒的离散化层数
    :param subdivisions: 球体的细分层级
    :return: 一个 Trimesh 对象(哑铃形状)

    作者: weiwei
    日期: 20191228大阪
    """
    stick = gen_rectstick(spos=spos, epos=epos, thickness=thickness, sections=sections)  # 创建棍棒部分
    spos_ball = gen_sphere(pos=spos, radius=thickness, subdivisions=subdivisions)  # 创建起始球
    epos_ball = gen_sphere(pos=epos, radius=thickness, subdivisions=subdivisions)  # 创建结束球
    vertices = np.vstack((stick.vertices, spos_ball.vertices, epos_ball.vertices))  # 合并顶点
    sposballfaces = spos_ball.faces + len(stick.vertices)  # 更新起始球的面索引
    endballfaces = epos_ball.faces + len(spos_ball.vertices) + len(stick.vertices)  # 更新结束球的面索引
    faces = np.vstack((stick.faces, sposballfaces, endballfaces))  # 合并面
    return trm.Trimesh(vertices=vertices, faces=faces)  # 返回哑铃形状的 Trimesh 对象


def gen_cone(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=0.005, sections=8):
    """
    生成一个锥体对象

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param radius: 锥体的底面半径,默认为 0.005 米
    :param sections: 用来近似圆柱体的离散化部分数
    :return: 一个 Trimesh 对象(锥体)

    作者: weiwei
    日期: 20191228大阪
    """
    height = np.linalg.norm(spos - epos)  # 计算锥体的高度
    pos = spos
    rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)  # 计算从 [0, 0, 1] 到目标方向的旋转矩阵
    homomat = rm.homomat_from_posrot(pos, rotmat)  # 计算旋转和平移的齐次变换矩阵
    return tp.Cone(height=height, radius=radius, sections=sections, homomat=homomat)  # 返回锥体对象


def gen_arrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, sticktype="rect"):
    """
    生成一个箭头对象(棍棒和锥体组成箭头)

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 箭头棍棒的厚度,默认为 0.005 米
    :param sections: 用来近似圆柱体的离散化部分数
    :param sticktype: 箭头棍棒末端的形状,圆形或矩形
    :return: 一个 Trimesh 对象(箭头)

    作者: weiwei
    日期: 20191228大阪
    """
    direction = rm.unit_vector(epos - spos)  # 计算箭头的方向
    stick = gen_stick(spos=spos, epos=epos - direction * thickness * 4, thickness=thickness, type=sticktype,
                      sections=sections)  # 创建箭头棍棒部分
    cap = gen_cone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)  # 创建箭头的锥体部分
    vertices = np.vstack((stick.vertices, cap.vertices))  # 合并顶点
    capfaces = cap.faces + len(stick.vertices)  # 更新锥体部分的面索引
    faces = np.vstack((stick.faces, capfaces))  # 合并面
    return trm.Trimesh(vertices=vertices, faces=faces)  # 返回箭头的 Trimesh 对象


def gen_dasharrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, lsolid=None, lspace=None,
                  sections=8, sticktype="rect"):
    """
    生成一个虚线箭头(由多段棍棒和箭头组成)

    :param spos: 起始点坐标,1x3 的 numpy 数组
    :param epos: 结束点坐标,1x3 的 numpy 数组
    :param thickness: 棍棒或箭头的厚度,默认为 0.005 米
    :param lsolid: 每段实线的长度,若为 None,则为 thickness 的 1 倍
    :param lspace: 每段空白的长度,若为 None,则为 thickness 的 1.5 倍
    :param sections: 棍棒近似圆形使用的段数
    :param sticktype: 棍棒的类型,"rect" 表示矩形截面
    :return: Trimesh 对象,表示虚线箭头的几何体

    作者: weiwei
    日期: 2019-12-28 大阪
    """
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)
    cap = gen_cone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)
    dash_stick = gen_dashstick(spos=spos,
                               epos=epos - direction * thickness * 4,
                               thickness=thickness,
                               lsolid=lsolid,
                               lspace=lspace,
                               sections=sections,
                               sticktype=sticktype)
    tmp_stick_faces = dash_stick.faces + len(cap.vertices)
    vertices = np.vstack((cap.vertices, dash_stick.vertices))
    faces = np.vstack((cap.faces, tmp_stick_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_axis(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=0.1, thickness=0.005):
    """
    生成一个三轴模型,带有箭头,表示坐标系(X、Y、Z)

    :param pos: 原点位置,1x3 numpy 数组
    :param rotmat: 旋转矩阵(每列为一个轴方向)
    :param length: 每个轴的长度
    :param thickness: 每个轴的棍棒厚度,默认0.005米
    :return: Trimesh 对象,表示一个三维坐标轴

    作者: weiwei
    日期: 2019-12-28 大阪
    """
    directionx = rotmat[:, 0]
    directiony = rotmat[:, 1]
    directionz = rotmat[:, 2]
    # x
    endx = directionx * length
    stickx = gen_stick(spos=pos, epos=endx, thickness=thickness)
    capx = gen_cone(spos=endx, epos=endx + directionx * thickness * 4, radius=thickness)
    # y
    endy = directiony * length
    sticky = gen_stick(spos=pos, epos=endy, thickness=thickness)
    capy = gen_cone(spos=endy, epos=endy + directiony * thickness * 4, radius=thickness)
    # z
    endz = directionz * length
    stickz = gen_stick(spos=pos, epos=endz, thickness=thickness)
    capz = gen_cone(spos=endz, epos=endz + directionz * thickness * 4, radius=thickness)

    # 合并所有几何体的顶点和面
    vertices = np.vstack(
        (stickx.vertices, capx.vertices, sticky.vertices, capy.vertices, stickz.vertices, capz.vertices))
    capxfaces = capx.faces + len(stickx.vertices)
    stickyfaces = sticky.faces + len(stickx.vertices) + len(capx.vertices)
    capyfaces = capy.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices)
    stickzfaces = stickz.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices) + len(capy.vertices)
    capzfaces = capz.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices) + len(
        capy.vertices) + len(stickz.vertices)
    faces = np.vstack((stickx.faces, capxfaces, stickyfaces, capyfaces, stickzfaces, capzfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              radius=0.1,
              thickness=0.005,
              sections=8,
              discretization=24):
    """
    生成一个圆环或环状箭头(圆环由多个小棒近似)

    :param axis: 圆环旋转轴(圆环围绕该轴旋转),1x3 numpy 数组
    :param starting_vector: 起始向量,决定圆环起点方向；默认与 axis 垂直
    :param portion: 圆环的比例(0.0 ~ 1.0),决定圆环占圆周的百分比
    :param center: 圆环中心位置,1x3 numpy 数组
    :param radius: 圆环半径
    :param thickness: 每段棒的厚度
    :param sections: 每个棒的离散段数,用于模拟圆柱
    :param discretization: 用多少个小棒来近似整个圆环
    :return: Trimesh 对象,表示生成的圆环

    作者: weiwei
    日期: 2020-06-02
    """
    unitaxis = rm.unit_vector(axis)
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unitaxis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)
    # 首先生成最后一秒
    # 生成剩余的环面
    if ndist > 0:
        # 初始化第一段
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  starting_vector) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), starting_vector) * radius
        stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
        vertices = stick.vertices
        faces = stick.faces

        # 继续添加其余段
        lastpos = starting_pos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), starting_vector) * radius
            stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        # 如果 portion 为 0,则返回空几何体
        return trm.Trimesh()

def gen_curveline(pseq, rotseq, r, section=5, toggledebug=False):
    """
    生成一个曲线线段的 Trimesh 对象

    :param pseq: 点序列,形状为 (N, 3)
    :param rotseq: 旋转序列,形状为 (N, 3, 3)
    :param r: 半径
    :param section: 每段的分段数
    :param toggledebug: 是否显示调试信息
    :return: 一个 Trimesh 对象(曲线线段)
    """
    vertices = []
    faces = []
    # 生成顶点
    for i, p in enumerate(pseq):
        for a in np.linspace(-np.pi, np.pi, section + 1):
            vertices.append(p + rotseq[i][:, 0] * r * np.sin(a)
                            + rotseq[i][:, 2] * r * np.cos(a))
    # 添加起始和结束点
    vertices.append(pseq[0])
    vertices.append(pseq[-1])
    # 生成面
    for i in range((section + 1) * (len(pseq) - 1)):
        if i % (section + 1) == 0:
            for v in range(i, i + section):
                faces.extend([[v, v + section + 1, v + section + 2],
                              [v, v + section + 2, v + 1]])
    # 生成封闭面
    for i in range(0, section):
        faces.extend([[i, len(vertices) - 2, i + 1]])
    for i in range(len(vertices) - section - 3, len(vertices) - 3):
        faces.extend([[i, len(vertices) - 1, i + 1]])
    # 调试信息
    if toggledebug:
        # show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
        # show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = gm.GeometricModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    # 创建最终的 Trimesh 对象
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
    objtrm.fix_normals()
    return objtrm


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=0.1,
                  thickness=0.005,
                  lsolid=None,
                  lspace=None,
                  sections=8,
                  discretization=24):
    """
    生成一个虚线圆环(环状由多段短棒构成)

    :param axis: 圆环将绕此轴旋转,1x3 numpy 数组
    :param portion: 圆环的比例(0.0 到 1.0 之间)
    :param center: 圆环的中心位置,1x3 numpy 数组
    :param radius: 圆环的半径
    :param thickness: 每段棍棒的粗细
    :param lsolid: 实线段的长度(若为空则自动根据厚度计算)
    :param lspace: 空白段的长度(若为空则自动根据厚度计算)
    :param sections: 用于近似圆柱形棒子的切段数
    :param discretization: 用于近似圆环的段数
    :return: Trimesh 对象,表示虚线圆环

    作者: weiwei
    日期: 2020年6月2日
    """
    assert (0 <= portion <= 1)
    solidweight = 1.6  # 默认实线段长度系数
    spaceweight = 1.07  # 默认空白段长度系数
    if not lsolid:
        lsolid = thickness * solidweight
    if not lspace:
        lspace = thickness * spaceweight

    unit_axis = rm.unit_vector(axis)
    starting_vector = rm.orthogonal_vector(unit_axis)

    # 计算最小离散化值,确保至少能分出一段虚线
    min_discretization_value = math.ceil(2 * math.pi / (lsolid + lspace))
    if discretization < min_discretization_value:
        discretization = min_discretization_value

    # 总共要生成多少段“实线”棍棒
    nsections = math.floor(portion * 2 * math.pi * radius / (lsolid + lspace))
    vertices = np.empty((0, 3))
    faces = np.empty((0, 3))
    for i in range(0, nsections):  # TODO wrap up end
        torus_sec = gen_torus(axis=axis,
                              starting_vector=rm.rotmat_from_axangle(axis, 2 * math.pi * portion / nsections * i).dot(
                                  starting_vector),
                              portion=portion / nsections * lsolid / (lsolid + lspace), center=center, radius=radius,
                              thickness=thickness, sections=sections, discretization=discretization)
        torus_sec_faces = torus_sec.faces + len(vertices)
        vertices = np.vstack((vertices, torus_sec.vertices))
        faces = np.vstack((faces, torus_sec_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=0.3,
                  center=np.array([0, 0, 0]),
                  radius=0.005,
                  thickness=0.0015,
                  sections=8,
                  discretization=24):
    """
    生成一个带箭头的圆环(通常用于表示旋转方向)

    :param axis: 圆环将绕此轴旋转,1x3 numpy 数组
    :param starting_vector: 起始方向向量(必须与 axis 垂直),若为空则自动生成
    :param portion: 圆环的比例(0.0 ~ 1.0)
    :param center: 圆环的中心点坐标
    :param radius: 圆环半径
    :param thickness: 棍棒和箭头的厚度
    :param sections: 圆柱切段数(越高越平滑)
    :param discretization: 用于近似圆形的段数
    :return: Trimesh 对象,表示这个带箭头的环

    作者: weiwei
    日期: 2020年6月2日
    """
    unitaxis = rm.unit_vector(axis)
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unitaxis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)

    if ndist > 0:
        # 最后一段是箭头
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  starting_vector) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), starting_vector) * radius
        arrow = gen_arrow(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, sticktype="round")
        vertices = arrow.vertices
        faces = arrow.faces

        # 生成前面的圆环段(棍棒)
        lastpos = starting_pos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), starting_vector) * radius
            stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        # 如果 portion = 0,不生成任何几何体
        return trm.Trimesh()


def facet_boundary(objtrimesh, facet, facetcenter, facetnormal):
    """
    计算一个面的边界多边形

    假设:
    1. 只有一个边界
    2. 面是凸的
    :param objtrimesh: Trimesh 数据类型
    :param facet: Trimesh 中定义的面数据类型
    :param facetcenter: 面的中心,用于计算变换
    :param facetnormal: 面的法向量,用于计算变换
    :return: 返回三个值:
        - [1x3 顶点列表,1x2 顶点列表,4x4 齐次变换矩阵]

    作者: weiwei
    日期: 20161213tsukuba
    """
    facetp = None
    # 使用 -facetnormal 使面朝下
    facethomomat = trm.geometry.plane_transform(facetcenter, -facetnormal)
    for i, faceidx in enumerate(facet):
        vert0 = objtrimesh.vertices[objtrimesh.faces[faceidx][0]]
        vert1 = objtrimesh.vertices[objtrimesh.faces[faceidx][1]]
        vert2 = objtrimesh.vertices[objtrimesh.faces[faceidx][2]]
        vert0p = rm.homotransformpoint(facethomomat, vert0)
        vert1p = rm.homotransformpoint(facethomomat, vert1)
        vert2p = rm.homotransformpoint(facethomomat, vert2)
        facep = shpg.Polygon([vert0p[:2], vert1p[:2], vert2p[:2]])
        if facetp is None:
            facetp = facep
        else:
            facetp = facetp.union(facep)
    verts2d = list(facetp.exterior.coords)
    verts3d = []
    for vert2d in verts2d:
        vert3d = rm.homotransformpoint(rm.homoinverse(facethomomat), np.array([vert2d[0], vert2d[1], 0]))[:3]
        verts3d.append(vert3d)
    return verts3d, verts2d, facethomomat


def extract_subtrimesh(objtrm, face_id_list, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    提取指定面ID的子网格

    :param objtrm: Trimesh 对象
    :param face_id_list: 面的ID列表
    :param offset_pos: 顶点位置的偏移(默认为[0, 0, 0])
    :param offset_rotmat: 顶点的旋转矩阵(默认为单位矩阵)
    :return: 返回一个新的 Trimesh 对象,表示提取出的子网格

    作者: weiwei
    日期: 20210120
    """
    if not isinstance(face_id_list, list):
        face_id_list = [face_id_list]
    tmp_vertices = offset_rotmat.dot(objtrm.vertices[objtrm.faces[face_id_list].flatten()].T).T + offset_pos
    tmp_faces = np.array(range(len(tmp_vertices))).reshape(-1, 3)
    return trm.Trimesh(vertices=tmp_vertices, faces=tmp_faces)


def extract_face_center_and_normal(objtrm, face_id_list, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    提取与面 ID列表对应的面中心数组和面法线数组

    如果 face_id_list 只有一个值,则返回单个法线和面中心
    :param objtrm: Trimesh 对象
    :param face_id_list: 面ID列表
    :param offset_pos: 顶点位置的偏移(默认为[0, 0, 0])
    :param offset_rotmat: 顶点的旋转矩阵(默认为单位矩阵)
    :return: 面中心数组和法线数组,若 face_id_list 只有一个元素,则返回单个值

    作者: weiwei
    日期: 20210120
    """
    return_sgl = False
    if not isinstance(face_id_list, list):
        face_id_list = [face_id_list]
        return_sgl = True
    # 计算面中心的偏移位置
    seed_center_pos_array = offset_rotmat.dot(
        np.mean(objtrm.vertices[objtrm.faces[face_id_list].flatten()], axis=1).reshape(-1, 3).T).T + offset_pos
    # 计算面法线的偏移
    seed_normal_array = offset_rotmat.dot(objtrm.face_normals[face_id_list].T).T
    if return_sgl:
        return seed_center_pos_array[0], seed_normal_array[0]
    else:
        return seed_center_pos_array, seed_normal_array


def gen_surface(surface_callback, rng, granularity=.01):
    """
    生成一个由网格数据构成的表面

    :param surface_callback: 用于计算网格顶点的回调函数
    :param rng: [[dim0_min, dim0_max], [dim1_min, dim1_max]] 表示生成表面的范围
    :param granularity: 网格的粒度(默认值为0.01)
    :return: 返回 Trimesh 对象,表示生成的表面

    作者: weiwei
    日期: 20210624
    """

    def _mesh_from_domain_grid(domain_grid, vertices):
        domain_0, domain_1 = domain_grid
        nrow = domain_0.shape[0]
        ncol = domain_0.shape[1]
        faces = np.empty((0, 3))
        for i in range(nrow - 1):
            # 定义网格三角形面片
            urgt_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
            urgt_pnt1 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
            urgt_pnt2 = np.arange(i * ncol + 1, i * ncol + ncol).T
            faces = np.vstack((faces, np.column_stack((urgt_pnt0, urgt_pnt2, urgt_pnt1))))
            blft_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
            blft_pnt1 = np.arange(i * ncol + ncol, i * ncol + ncol + ncol - 1).T
            blft_pnt2 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
            faces = np.vstack((faces, np.column_stack((blft_pnt0, blft_pnt2, blft_pnt1))))
        return trm.Trimesh(vertices=vertices, faces=faces)

    # 计算网格的大小和粒度
    a_min, a_max = rng[0]
    b_min, b_max = rng[1]
    n_a = round((a_max - a_min) / granularity)
    n_b = round((b_max - b_min) / granularity)

    # 生成网格
    domain_grid = np.meshgrid(np.linspace(a_min, a_max, n_a, endpoint=True),
                              np.linspace(b_min, b_max, n_b, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel(), domain_1.ravel()))
    # 使用回调函数计算每个点的高度值
    codomain = surface_callback(domain)
    vertices = np.column_stack((domain, codomain))
    return _mesh_from_domain_grid(domain_grid, vertices)


if __name__ == "__main__":
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .2, .3], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    # objcm = gm.WireFrameModel(gen_torus())
    # objcm.set_rgba([1, 0, 0, 1])
    # objcm.attach_to(base)
    # objcm = gm.StaticGeometricModel(gen_axis())
    # objcm.set_rgba([1, 0, 0, 1])
    # objcm.attach_to(base)

    import time

    tic = time.time()
    for i in range(100):
        gen_dumbbell()
    toc = time.time()
    print("mine", toc - tic)
    objcm = gm.GeometricModel(gen_dashstick(lsolid=.005, lspace=.005))
    objcm = gm.GeometricModel(gen_dashtorus(portion=1))
    objcm.set_rgba([1, 0, 0, 1])
    objcm.attach_to(base)

    objcm = gm.GeometricModel(gen_stick())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, .01, 0]))
    objcm.attach_to(base)

    objcm = gm.GeometricModel(gen_dasharrow())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, -.01, 0]))
    objcm.attach_to(base)
    base.run()
