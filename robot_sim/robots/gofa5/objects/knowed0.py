import math
import os

import shapely
from fontTools.misc.plistlib import end_key
from shapely.geometry import Polygon
from shapely.geometry import Point
from direct.task.TaskManagerGlobal import taskMgr

import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.GOFA5.gofawithag as gf5  ## gofowithag 将爪子加入了cc
import modeling.collision_model as cm
import numpy as np
import grasping.planning.antipodal as gpa
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.ag145.ag145 as ag
import drivers.devices.dh.ag145 as dh_r
import robot_con.gofa_con.gofa_con as gofa_con
import basis.trimesh as trimeshWan
import humath
import copy

from onepic import *


##当完成一次物体转换之后,写一个中断函数；打印信息,回车进行拍照代码...

##全部在z正半轴:
def placeGround(obj, displacement_homo_list):
    for homo in displacement_homo_list:
        obj_copy = obj.copy()
        z_max = 0
        for vertice in obj_copy.objtrm.vertices:
            wd_vertice = [vertice[0], vertice[1], vertice[2], 1]
            wd_vertice = np.dot(homo, wd_vertice)
            distance = -wd_vertice[2]
            # print("距离？", distance)
            if distance >= z_max:
                z_max = distance
        homo[2, 3] = homo[2, 3] + z_max - 0.014  ##.  i=3时  设为0.05
    return displacement_homo_list


def ava_space(obj_poslist, step=0.04, slight_center=[0.54, 0, 0]):
    ##可用的空间  z默认为0
    ava_space = []
    for i in range(-5, 6, 1):
        for j in range(-5, 4, 1):
            [x, y, z] = [slight_center[0] + step * i, slight_center[1] + step * j, slight_center[2]]
            [x, y, z] = np.round([x, y, z], 2)
            ava_space.append([x, y, z])
    for obj_xyz in obj_poslist:
        ava_space.remove(obj_xyz)
    return ava_space


def translate_pic(point_table):
    # 我在旋转图像,图像在动,说明T是图像相对于桌面的
    Trans = rm.homomat_from_posrot(pos=np.array([0.2, -0.64, 1.42]),
                                   rot=rm.rotmat_from_euler(ai=math.degrees(-154), aj=math.degrees(7.5), ak=0))
    print("变换矩阵:", Trans)
    Trans_inv = np.linalg.inv(Trans)
    point_pic = math.dot(Trans_inv, point_table)
    print(point_pic)
    return point_pic


def update_obj_poslist(obj_list):
    new_obj_poslist = []
    for obj in obj_list:
        obj_pos = obj.get_pos()
        obj_pos[2] = 0
        obj_pos = np.round(obj_pos, 2)
        obj_pos = obj_pos.tolist()
        new_obj_poslist.append(obj_pos)
    print("新的obj列表？", new_obj_poslist)
    return new_obj_poslist


def rand_xyz(obj, step=0.08, slight_center=[0.54, 0, 0], ):
    rand_int = np.random.randint(-5, 5, 2)
    rand_x = slight_center[0] + rand_int[0] * step
    rand_y = slight_center[0] + rand_int[1] * step
    z = obj.get_pos()[2]
    return np.array([rand_x, rand_y, z])


def grasper_plannning(obj, angle_between_contact_normals=math.radians(177), max_samples=5):
    global gripper_s
    gripper_s = ag.Ag145()
    grasp_info_list = gpa.plan_grasps(gripper_s, obj, angle_between_contact_normals,
                                      openning_direction="loc_y",
                                      max_samples=max_samples,
                                      min_dist_between_sampled_contact_points=0.003,
                                      contact_offset=0.003)
    return grasp_info_list


##  凸包面的计算
def getFacetsCenter(obj_trimesh, facets):
    '''
    get the coordinate of large facet center in self.largefacetscenter
    get the normal vecter of large facet center in self.largefacet_normals
    get the vertices coordinate of large facet in self.largefacet_vertices
    :return:
    '''
    vertices = obj_trimesh.vertices
    faces = obj_trimesh.faces
    facets = facets  ##[list([14, 15]) list([1, 5]) list([11, 13]) list([9, 10, 6]) list([0, 2])
    ## list([3, 4, 12]) list([8, 7])]
    smallfacesarea = obj_trimesh.area_faces  ##16个面积
    smallface_normals = obj_trimesh.face_normals
    smallfacecenter = []
    # prelargeface = []
    smallfacectlist = []
    smallfacectlist_area = []
    largefacet_normals = []
    largefacet_vertices = []
    largefacetscenter = []
    for i, smallface in enumerate(faces):  ###     计算面的 中心坐标
        smallfacecenter.append(humath.centerPoint(np.array([vertices[smallface[0]],  ## 第0个顶点的坐标
                                                            vertices[smallface[1]],  ##第1个顶点的坐标
                                                            vertices[smallface[2]]])))  ##第3个顶点的坐标

    ##watch smallfacecenter
    # print("all small face center", smallfacecenter)  ## len==16 16 个面
    # point=gm.gen_sphere(radius=0.006,pos=smallfacecenter[15])
    # point.set_rgba([0,1,0,1])
    # point.attach_to(base)
    prelargeface = copy.deepcopy(facets)
    for facet in facets:  ##list[14,15]  三点面的序号
        b = []
        b_area = []
        temlargefaceVerticesid = []
        temlargefaceVertices = []
        for face in facet:
            b.append(smallfacecenter[face])  ## b存放的该面的中心坐标
            b_area.append(smallfacesarea[face])  ##存放该点的面积
            temlargefaceVerticesid.extend(faces[face])  ## 存放面序号为14 15的 里面的顶点序号
            # print("temlargefaceVerticesid", temlargefaceVerticesid)
        smallfacectlist.append(b)  # 序号为14 15 的面的中心点
        smallfacectlist_area.append(b_area)
        smallfacenomallist = [smallface_normals[facet[j]] for j in range(len(facet))]  ##这俩/三的法向量保存
        largefacet_normals.append(np.average(smallfacenomallist, axis=0))  ##以列的取平均
        # self.largefacet_normals.append(self.smallface_normals[facet[0]]) #TODO an average normal
        temlargefaceVerticesid = list(set(temlargefaceVerticesid))  # remove repeating vertices ID
        for id in temlargefaceVerticesid:  ## 完整的一个面的定点序号
            temlargefaceVertices.append(vertices[id])  ## 记录4个顶点确定一个面的4个坐标
        largefacet_vertices.append(temlargefaceVertices)  ##记录所有面下的四个坐标【面0的4坐标】【面1的4坐标】...
    for i, largeface in enumerate(smallfacectlist):  ##7个面的中心点坐标(包含重叠部分的)
        largefacetscenter.append(humath.centerPointwithArea(largeface, smallfacectlist_area[i]))  ##利用加权思想计算重叠面的质心
    return largefacetscenter, largefacet_normals, largefacet_vertices


##6个面的长方体
def rand_changfangti(init_obj, rand_pos):
    """
    躺下/倾斜----旋转
    """
    all_rotmat = []
    all_rotmat.append(np.eye(3))
    rot_axis = [[1, 0, 0], [0, 1, 0]]
    rot_angel = [math.pi / 2, math.pi, 3 * math.pi / 2]
    for axis in rot_axis:
        for angel in rot_angel:
            rotmatxy = rm.rotmat_from_axangle(axis=axis, angle=angel)
            print("单独的？:", rotmatxy)
            all_rotmat.append(rotmatxy)
    print("所有的", all_rotmat)
    i = np.random.randint(len(all_rotmat))
    print("选中了i", i)
    ##设置关于世界z轴旋转:
    angel_z = np.random.uniform(0, 2 * math.pi)
    print("绕世界z旋转:", angel_z)
    rotmat = np.dot(rm.rotmat_from_axangle(axis=[0, 0, 1], angle=angel_z), all_rotmat[i])
    print(rotmat)
    obj = init_obj.copy()
    obj.set_pos(rand_pos)
    obj.set_rotmat(rotmat)
    obj.set_rgba([0, 1, 0, 0.5])
    obj.attach_to(base)
    gm.gen_frame(rand_pos, obj.get_rotmat(), length=0.15).attach_to(base)
    return obj


def obj_stablepose(obj):
    facet_project_list = []
    com_project_list = []
    stable_ids = []
    obj_ch = obj.objtrm.convex_hull
    vertices = obj_ch.vertices
    faces = obj_ch.faces
    com = obj.objtrm.center_mass

    convex_obj = trimeshWan.Trimesh(vertices=vertices, faces=faces)

    ##displacement
    # convex_obj_gm=gm.GeometricModel(convex_obj)
    # convex_obj_gm.set_rgba([1,1,0,0.5])
    # convex_obj_gm.attach_to(base)

    facets, facetnormals, facetcurvatures = convex_obj.facets_noover(faceangle=0.99)
    facets_center, facet_normals, facet_vertices = getFacetsCenter(convex_obj, facets)  ##基于物体在(0,0,0)位置的中心

    pos_list = [obj.get_pos() + center for center in facets_center]

    ##建立每个面的变换矩阵homo(坐标系),以法向量为z轴,中点为原点
    rotmat_list = [rm.rotmat_between_vectors(normal, np.array([0, 0, 1])) for normal in facet_normals]

    ##稳定pose检测:
    for id, facet in enumerate(facet_vertices):  ##facet:是 顶点坐标 构成的  面  .num(id)=7   基于原点的坐标
        # print("see facet",facet)
        facet_project = [rotmat_list[id].dot(vertex)[:2] for vertex in facet]  ## 物体发生旋转之后的 顶点坐标  在投影
        facet_project_list.append(facet_project)
        com_project_list.append(rotmat_list[id].dot(com)[:2])  ## com点也要投影

    for id, facet_project in enumerate(facet_project_list):  ##每个面  发生旋转之后的顶点坐标【x,y】

        contact_polygon = Polygon(facet_project)  ## polygon构建多边形  储存的是多边形的坐标
        # print("contact_polygon",contact_polygon)
        convex_hull_polygon = contact_polygon.convex_hull  ##  利用坐标  构建凸包多边形！！
        # #可视化的过程  监测
        # contact_patch = PolygonPatch(convex_hull_polygon, fc='yellow', ec='black', alpha=0.5)   ##可视化多边形的接触块.
        # ##alpha 是透明度.   ec是边框颜色
        # fig = plt.figure(figsize=(5, 5), dpi=100)
        # plt.axis('on')
        # # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.set_ylim(-0.2, 0.2)
        # ax1.set_xlim(-0.2, 0.2)
        # # ring_patch2 = PolygonPatch(contact_patch, color="yellow", alpha=0.5)
        # ax1.add_patch(contact_patch)   ##将凸包多边形 增加到图中

        if shapely.within(Point(com_project_list[id][0], com_project_list[id][1]), convex_hull_polygon):
            # ax1.scatter(x=com_project_list[id][0], y=com_project_list[id][1], color="green")
            stable_ids.append(id)
    homo_list = [rm.homomat_from_posrot(pos_list[i], rotmat_list[i]) for i in stable_ids]
    homo_inv_list = [np.linalg.inv(homo) for homo in homo_list]

    return homo_list, homo_inv_list


def go_init():
    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_jnts = rg_s.get_jnt_values("arm")
    path = rrtc_s.plan(component_name="arm",
                       start_conf=current_jnts,
                       goal_conf=init_jnts,
                       ext_dist=0.03,
                       max_time=3500)
    # rbt_r.move_jntspace_path(path)


def Return_Path(pos, init_posset, robot, conf_list):
    return_start_conf = conf_list[-1]
    min_dd = np.inf
    for i in init_posset:
        dd = np.linalg.norm(i - pos, 2)
        if dd <= min_dd:
            min_dd = dd
            init_pos = i
    return_start_conf = conf_list[-1]
    init_rotmat = robot.get_gl_tcp(manipulator_name="arm")[1]
    return_end_conf = rg_s.ik(component_name="arm", tgt_pos=init_pos, tgt_rotmat=init_rotmat)
    rrtc_planner = rrtc.RRTConnect(rg_s)
    path = rrtc_planner.plan(component_name="arm", goal_conf=return_end_conf, start_conf=return_start_conf,
                             obstacle_list=obstacle_list,
                             ext_dist=0.05, max_iter=50, smoothing_iterations=50, max_time=500)
    return path


if __name__ == '__main__':
    # np.random.seed(0)

    base = wd.World(cam_pos=[2.554, 2.5, 2.5], lookat_pos=[0, 0, 0])  ##看向的位置
    gm.gen_frame().attach_to(base)

    gripper_s = ag.Ag145()
    # gripper_r = dh_r.Ag145driver()
    # gripper_r.init_gripper()
    rg_s = gf5.GOFA5()
    # rbt_r = gofa_con.GoFaArmController()
    rrtc_s = rrtc.RRTConnect(rg_s)
    ppp_s = ppp.PickPlacePlanner(rg_s)

    # cd_conf=np.array([ 0.25295786,  0.86826844,  0.06268809, -0.88155852,  1.09102748,-0.70973774])
    # rg_s.fk(component_name="arm",jnt_values=cd_conf)
    # rg_s.gen_meshmodel(toggle_tcpcs=True,rgba=[1,0,0,1]).attach_to(base)
    # check,cpoint=rg_s.is_collided(obstacle_list=[],toggle_contact_points=True)
    # print("检查碰撞",check,cpoint)
    # if check:
    #     for i in cpoint:
    #      gm.gen_frame(pos=i ,length=0.2).attach_to(base)
    # base.run()
    # go_init()
    # base.run()

    ##定义目标位置 和视线
    # slight = gm.movegmbox(extent=[0.8, 0.8, 0.005], rgba=[1, 0, 0, 0.6], pos=[0.54, 0, 0.01])  extent---总长 , 坐标为一半
    ##
    # slight.attach_to(base)
    slight_center = [0.54, 0, 0]
    gm.gen_frame(slight_center).attach_to(base)

    ## 测试一下slight的四个点坐标
    ball1 = cm.gen_sphere(radius=0.01, rgba=[1, 0, 0, 1])
    ball1.set_pos(np.array([0.14, -0.4, 0]))
    ball1.attach_to(base)

    ball2 = cm.gen_sphere(radius=0.01, rgba=[0, 1, 0, 1])
    ball2.set_pos(np.array([0.14, 0.4, 0]))
    ball2.attach_to(base)

    ball3 = cm.gen_sphere(radius=0.01, rgba=[0, 0, 1, 1])
    ball3.set_pos(np.array([0.94, -0.4, 0]))
    ball3.attach_to(base)

    ball4 = cm.gen_sphere(radius=0.01, rgba=[1, 1, 1, 1])
    ball4.set_pos(np.array([0.94, 0.4, 0]))
    ball4.attach_to(base)

    ##机器人
    # rg_s=gf5.GOFA5()
    manipulator_name = "arm"
    hand_name = "hnd"
    gm.gen_frame(rg_s.get_gl_tcp("hnd")[0], rg_s.get_gl_tcp("hnd")[1]).attach_to(base)
    rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    rg_s.show_cdprimit()
    # rrtc_s=rrtc.RRTConnect(rg_s)
    # ppp_s=ppp.PickPlacePlanner(rg_s)
    start_conf = np.array([0, 0, 0, 0, 0, 0])

    ## 物体obj_ti
    init_posobj = ava_space(obj_poslist=[]).copy()  ##whole space
    print("可用的空间有几个？:", len(init_posobj))
    print("可以用的空间:", init_posobj)
    # for init_pos in init_posobj:
    #     init_pos[2]=0.005
    # print(init_posobj)

    this_dir, this_filename = os.path.split(__file__)
    obj_ti = cm.CollisionModel(r"D:\wrs-fujikoshi-main-1217\0000_placementplanner\objects\tixing.STL")
    obj_ti.change_cdprimitive_type(cdprimitive_type='surface_balls', expand_radius=0.0008)
    obj_ti.set_rgba([1, 1, 0, 1])

    # obj_ti.attach_to(base)
    # base.run()

    # 读取爪子:
    # gripper_s = ag.Ag145()
    grasp_info_list = gpa.load_pickle_file(objcm_name="tixing", root=None,
                                           file_name="objti_grasptest_250116.pickle")  ##570
    # print(len(grasp_info_list))    ##---box0  274
    # base.run()

    max_eps = 2
    i_ep = 0
    last_start_homo = None
    last_obj = None
    r_path = []
    jaw_path = []
    obj_path = []

    init_posset = []
    init_pos0 = np.array([0.3, 0.5, 0.2])
    init_pos1 = np.array([0.2, -0.5, 0.2])
    init_posset.append(init_pos0)
    init_posset.append(init_pos1)

    # # 文件轨迹
    # obj_tiall = []
    # ##计算all poses:
    # homo_list,homo_inv_list=obj_stablepose(obj_ti)
    # pos_id_list=[49,57,69,83,95,74]
    # pos_list=[init_posobj[i] for i in pos_id_list]
    # displacement_homo_list = [rm.homomat_from_posrot(pos_list[i]).dot(homo_inv_list[i]) for i in
    #                               range(len(homo_list))]
    # new_displacement_homo_list=placeGround(obj_ti,displacement_homo_list)
    # for homo in new_displacement_homo_list:
    #     obj_copy=obj_ti.copy()
    #     obj_copy.set_rgba([0,1,1,0.5])
    #     obj_copy.set_homomat(homo)
    #     # obj_copy.attach_to(base)
    #     obj_tiall.append(obj_copy)
    # # obstacle_list=[]

    # 已知obj0的状态
    obj0 = obj_ti.copy()
    obj0.set_homomat(np.array([[0.80000007, -0.39999995, 0.44721353, 0.54329181],
                               [-0.39999995, 0.19999999, 0.89442718, -0.07841641],
                               [-0.44721353, -0.89442718, 0., 0.05308204],
                               [0., 0., 0., 1.]]))

    # obj0.attach_to(base)
    # 物体0的位置在哪里？？ [0.54329181 - 0.07841641  0.05308204]
    # obj0.show_cdmesh()
    # obstacle_list.append(obj0)
    # check,cpoint=rg_s.is_collided(obstacle_list=obstacle_list,toggle_contact_points=True)
    # print("检查碰撞",check,cpoint)
    # if check:
    #     for i in cpoint:
    #      gm.gen_frame(pos=i ,length=0.2).attach_to(base)
    # base.run()

    #
    conf_list = None
    while conf_list == None:
        obj_tiall = []
        obstacle_list = []
        ##计算all poses:
        homo_list, homo_inv_list = obj_stablepose(obj_ti)
        # print("凸面数量",len(homo_list))
        # print("看看stable-homo-inv-list:",homo_inv_list)

        # # 随机一个homo_inv
        # start_homo=homo_list[5]
        # spos=start_homo[:3,3]
        # srotmat=start_homo[:3,:3]
        # gm.gen_frame(spos,srotmat).attach_to(base)
        # starthomo_inv=homo_inv_list[5]

        ##pos的个数跟着  凸面 个数而定
        pos_id_list = [49, 57, 69, 83, 95, 74]

        pos_list = [init_posobj[i] for i in pos_id_list]
        # base.run()

        displacement_homo_list = [rm.homomat_from_posrot(pos_list[i]).dot(homo_inv_list[i]) for i in
                                  range(len(homo_list))]
        new_displacement_homo_list = placeGround(obj_ti, displacement_homo_list)
        # print("new homo length",len(new_displacement_homo_list))

        for homo in new_displacement_homo_list:
            obj_copy = obj_ti.copy()
            obj_copy.set_rgba([0, 1, 1, 0.5])
            obj_copy.set_homomat(homo)
            # obj_copy.attach_to(base)
            obj_tiall.append(obj_copy)
        # base.run()

        # dex0,dex1,dex2=np.random.choice(len(pos_id_list),size=3,replace=False)
        # print("选择的dex ",dex0,dex1,dex2)
        # #pick  green
        # obj0=obj_tiall[0].copy()
        # obj0.set_rgba([0,1,0,1])

        # gm.gen_frame(obj0.get_pos(),obj0.get_rotmat()).attach_to(base)
        # obj0.show_cdprimit()

        ##place     small red
        # obj00 = obj0.copy()
        while True:

            dex0, dex1, dex2 = np.random.choice(len(pos_id_list), size=3, replace=False)
            print("选择的dex:", dex0, dex1, dex2)
            obj00 = obj_tiall[dex1].copy()
            obj00_state = np.round(obj00.get_rotmat(), 3)
            obj0_state = np.round(obj0.get_rotmat(), 3)
            if not np.array_equal(obj00_state, obj0_state):
                print(obj00_state, obj0_state)
                break

        obj00 = obj_tiall[dex1].copy()
        # obj00.attach_to(base)
        # base.run()
        ##replace:  blue
        obj000 = obj_tiall[dex2].copy()
        obj000.set_rgba([0, 0, 1, 0.8])

        # ##观察3个位置
        # obj0.attach_to(base)
        # obj000.attach_to(base)
        # obj00.attach_to(base)
        # base.run()

        # check,cpoint=rg_s.is_collided(obstacle_list=obstacle_list,toggle_contact_points=True)
        #     print("检查碰撞",check,cpoint)
        #     if check:
        #         for i in cpoint:
        #          gm.gen_frame(pos=i ,length=0.2).attach_to(base)
        #     ccp= [np.array([ 0.53038633, -0.08748391, -0.12842003]), np.array([ 0.57513672, -0.09080687, -0.12398088]), np.array([ 0.58017659, -0.05717814, -0.02951322]), np.array([ 0.61086911, -0.05008442, -0.02347497]), np.array([0.64176929, 0.02058455, 0.00210524]), np.array([ 0.6444816 ,  0.01263568, -0.03405447]), np.array([ 0.6506018 ,  0.01290834, -0.03665378])]
        #     for c in ccp:
        #         gm.gen_frame(pos=c, length=0.2).attach_to(base)
        #
        #     base.run()
        start_homo = np.array(obj0.get_homomat())

        # # gm.gen_frame(start_pos,start_rotmat).attach_to(base)##   末端执行器和物体是不一样的
        end_homo = np.array(obj00.get_homomat())
        # if i_ep>0:
        #     obj0=last_obj.copy()
        #     obj00=obj000.copy()
        #     print(">>>")
        #     start_homo=last_start_homo
        #     end_homo=np.array(obj000.get_homomat())

        # ##检查是否能达到  start  和  end
        #     jnt_values=rg_s.ik(component_name="arm",tgt_pos=start_pos,tgt_rotmat=start_rotmat)
        #     rg_s.fk(component_name="arm",jnt_values=jnt_values)   ## start 能到到
        #     rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
        # base.run()
        #
        #
        #     jnt_values=rg_s.ik(component_name="arm",tgt_pos=end_pos,tgt_rotmat=end_rotmat)
        #     rg_s.fk(component_name="arm",jnt_values=jnt_values)   ## end 能到到
        #     rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
        # #     base.run()
        obstacle_list = []
        conf_list, jawwidth_list, objpose_list = \
            ppp_s.gen_pick_and_place_motion(hnd_name="arm",
                                            objcm=obj0,
                                            grasp_info_list=grasp_info_list,
                                            start_conf=start_conf,
                                            end_conf=None,
                                            goal_homomat_list=[start_homo, end_homo],
                                            approach_direction_list=[None, np.array([0, 0, -1])],
                                            approach_distance_list=[.1] * 2,
                                            depart_direction_list=[np.array([0, 0, 1]), None],
                                            obstacle_list=obstacle_list,
                                            depart_distance_list=[.1] * 2)
        if conf_list == None:
            print("重新选择可切换的状态")
    obj0.set_rgba([1, 0, 0, 1])
    obj00.set_rgba([0, 1, 0, 1])
    obj0.attach_to(base)
    obj00.attach_to(base)
    print("物体0的位置在哪里？？", obj0.get_pos())
    print("物体00的位置在哪里？？", obj00.get_pos())
    # print("看看conf",conf_list)

    # # base.run()
    #  ##保存规划之后的路径.
    #  #    with open("test_conf_list0212004", "wb") as f:
    #  #        pickle.dump(conf_list, f)
    #  #    with open("test_jawwidth_path0212004", "wb") as f:
    #  #        pickle.dump(jawwidth_list, f)
    #  #    with open("test_objpose_list0212004", "wb") as f:
    #  #        pickle.dump(objpose_list, f)
    #  #
    #  #
    #  #    with open("test_conf_list0212004","rb") as f:
    #  #        conf_list=pickle.load(f)
    #  #    with open("test_jawwidth_path0212004","rb") as f:
    #  #        jawwidth_list=pickle.load(f)
    #  #    with open("test_objpose_list0212004","rb") as f:
    #  #        objpose_list=pickle.load(f)
    #  #    print("长度是？？",len(conf_list))
    #  #    print("看看conf",conf_list)
    #
    #     # 转换obj之后,回到start pose   利用rrt   扩展jaw pose obj

    return_path = Return_Path(obj00.get_pos(), init_posset=init_posset, robot=rg_s, conf_list=conf_list)
    robot_paths = []
    jawwidth_paths = []
    objposes_list = []
    path_seg_id = [0]
    for i in range(len(conf_list) - 1):
        if jawwidth_list[i] != jawwidth_list[i - 1]:
            path_seg_id.append(i)
            path_seg_id.append(i + 1)
            # path_seg_id.append(i + 2)
    path_seg_id.append(len(conf_list))

    for i in range(len(path_seg_id) - 1):
        robot_paths.append(conf_list[path_seg_id[i]:path_seg_id[i + 1]])
        jawwidth_paths.append(jawwidth_list[path_seg_id[i]:path_seg_id[i + 1]])
        objposes_list.append(objpose_list[path_seg_id[i]:path_seg_id[i + 1]])

    robot_paths.append(return_path)
    jawwidth_paths.append([jawwidth_list[-1]] * len(return_path))
    objposes_list.append([objpose_list[-1]] * len(return_path))

    #
    # ##拍照的状态
    # robot_paths.append([return_path[-1]])
    # jawwidth_paths.append([jawwidth_list[-1]])
    # objposes_list.append([objpose_list[-1]])
    # #
    # # ##第一次转换的结束状态---作为第二次的开始
    # # last_start_homo=end_homo
    # # last_obj=obj00.copy()
    # # last_obj.set_rgba([0,1,0,0.5])   ##small green
    # #
    # # r_path.append(robot_paths)
    # # jaw_path.append(jawwidth_paths)
    # # obj_path.append(objposes_list)
    # # i_ep+=1
    import pickle

    # 读取文件
    with open("test_conf_list0226", "wb") as f:
        pickle.dump(robot_paths, f)
    with open("test_jawwidth_path0226", "wb") as f:
        pickle.dump(jawwidth_paths, f)
    with open("test_objpose_list0226", "wb") as f:
        pickle.dump(objposes_list, f)

    with open("test_conf_list0226", "rb") as f:
        robot_paths = pickle.load(f)
    with open("test_jawwidth_path0226", "rb") as f:
        jawwidth_paths = pickle.load(f)
    with open("test_objpose_list0226", "rb") as f:
        objposes_list = pickle.load(f)

    robot_attached_list = []
    object_attached_list = []
    counter = [0, 0]  ## 【第一次转换下,第0个轨迹, 该轨迹下第0个动作】
    print(objposes_list)


    # base.run()
    def update(robot_s,
               object_box,
               robot_paths,
               jawwidth_paths,
               obj_paths,
               robot_attached_list,
               object_attached_list,
               counter,
               task):

        if counter[0] >= len(robot_paths):
            print("拍照！！")

            counter[0] = 0

        if counter[1] >= len(robot_paths[counter[0]]):
            counter[1] = 0

        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()

            robot_attached_list.clear()
            object_attached_list.clear()

        pose = robot_paths[counter[0]][counter[1]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(hand_name, jawwidth_paths[counter[0]][counter[1]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_paths[counter[0]][counter[1]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        # print("jnts = ,", rbt_r.get_jnt_values())
        # print("torque = ,", rbt_r.get_torques())
        counter[1] += 1

        # if counter[1]==len(robot_paths[counter[0]]):
        if base.inputmgr.keymap["space"] is True:
            if len(robot_paths[counter[0]]) <= 2:
                # gripper_r.jaw_to(jawwidth_paths[counter[0]][0]*0.6)
                counter[0] += 1
                counter[1] = 0

            else:
                # rbt_r.move_jntspace_path(robot_paths[counter[0]], wait=True)
                counter[0] += 1
                counter[1] = 0

        return task.again


    taskMgr.doMethodLater(0.05, update, "update",
                          extraArgs=[rg_s,
                                     obj0,
                                     robot_paths,
                                     jawwidth_paths,
                                     objposes_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
    # 物体0的位置在哪里？？ [0.54329181 - 0.07841641  0.05308204]
    # 物体00的位置在哪里？？ [0.72500002 - 0.015       0.07600001]
