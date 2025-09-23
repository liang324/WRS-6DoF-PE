import numpy as np
import copy
import open3d as o3d
import sklearn.cluster as skc
import vision.depth_camera.pcd_data_adapter as pda


def __draw_registration_result(source_o3d, target_o3d, transformation):
    """
    可视化配准结果

    :param source_o3d: 源点云的 Open3D 对象
    :param target_o3d: 目标点云的 Open3D 对象
    :param transformation: 用于将源点云变换到目标点云的变换矩阵
    """
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])  # 源点云颜色

    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 目标点云颜色
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def __preprocess_point_cloud(pcd, voxel_size):
    """
    预处理点云,包括降采样和特征计算

    :param pcd: 输入的点云
    :param voxel_size: 用于降采样的体素大小
    :return: 降采样后的点云和其 FPFH 特征
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    down_radius_normal = voxel_size * 3
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
    radius_feature = voxel_size * 3
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def registration_ptpt(src, tgt, downsampling_voxelsize=.003, toggledebug=False):
    """
    使用全局配准和局部 ICP 配准两个点云

    :param src: 源点云的 numpy 数组
    :param tgt: 目标点云的 numpy 数组
    :param downsampling_voxelsize: 降采样的体素大小
    :param toggledebug: 是否开启调试模式
    :return: 配准质量和变换矩阵,质量以所有内点对应关系的 RMSE 衡量

    author: ruishuang, revised by weiwei
    date: 20191210
    """
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))
    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)
    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: 下采样点云的RANSAC配准")
        print("   由于下采样体素大小为 %.3f," % downsampling_voxelsize)
        print("   我们使用自由的距离阈值 %.3f." % distance_threshold)
    # result_global = o3d.pipelines.registration.registration_fass_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    result_global = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)
    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: 点对点Open3D_ICP_Edition注册应用于原始点")
        print("  云来细化对齐,这一次我们使用了严格的")
        print("   距离阈值 %.3f" % distance_threshold)
    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, result_global.transformation, toggledebug=toggledebug)

    # def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    #     """
    #
    #     :param src:
    #     :param tgt:
    #     :param maxcorrdist:
    #     :param toggledebug:
    #     :return:
    #
    #     author: weiwei
    #     date: 20191229
    #     """
    #
    #     criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                        # converge if fitnesss smaller than this
    #                                                        relative_rmse=1e-6,  # converge if rmse smaller than this
    #                                                        max_iteration=2000)
    #     result_icp = o3d.pipelines.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    #     if toggledebug:
    #         __draw_registration_result(src, tgt, result_icp.transformation)
    #     return [result_icp.inlier_rmse, result_icp.transformation]
    #
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False))
    # if toggledebug:
    #     __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    # return [result_icp.inlier_rmse, result_icp.transformation]


def registration_ptpln(src, tgt, downsampling_voxelsize=2, toggledebug=False):
    """
    使用全局配准(RANSAC 基于特征匹配)和局部 ICP(点到面)配准两个点云

    :param src: 源点云 (numpy.ndarray),形状为 (N, 3)
    :param tgt: 目标点云 (numpy.ndarray),形状为 (M, 3)
    :param downsampling_voxelsize:  下采样体素大小 (float),用于减少点云密度,加快配准速度.默认为 2
    :param toggledebug:  是否开启调试模式 (bool),如果为 True,则会显示配准过程中的中间结果.默认为 False
    :return: list: 包含配准结果的列表,格式为 [inlier_rmse, transformation]
             - inlier_rmse (float):  内点均方根误差,衡量配准质量
             - transformation (numpy.ndarray):  4x4 齐次变换矩阵,将源点云变换到目标点云坐标系

    author: ruishuang, revised by weiwei
    date: 20191210
    """
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)

    def __preprocess_point_cloud(pcd, voxel_size):
        original_radius_normal = voxel_size
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=original_radius_normal, max_nn=30))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        down_radius_normal = voxel_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))
    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)
    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)
    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: Point-to-plane Open3D_ICP_Edition registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
    result_icp = o3d.pipelines.registration.registration_icp(
        src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.transformation]


def registration_icp_ptpt(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """
    使用 ICP(点到点)配准两个点云

    :param src: 源点云 (numpy.ndarray),形状为 (N, 3)
    :param tgt: 目标点云 (numpy.ndarray),形状为 (M, 3)
    :param inithomomat:  初始变换矩阵 (numpy.ndarray),4x4 齐次变换矩阵,作为 ICP 的初始估计.默认为单位矩阵
    :param maxcorrdist:  最大对应距离 (float),用于过滤不合理的对应关系.默认为 2
    :param toggledebug:  是否开启调试模式 (bool),如果为 True,则会显示配准过程中的中间结果.默认为 False
    :return:  与 `_registration_icp_ptpt_o3d` 函数的返回值相同

    author: weiwei
    date: 20191229
    """
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, inithomomat, maxcorrdist, toggledebug)


def remove_outlier(src_nparray, downsampling_voxelsize=2, nb_points=7, radius=3, estimate_normals=False,
                   toggledebug=False):
    """
    下采样并统计移除离群点

    :param src_nparray:  源点云 (numpy.ndarray),形状为 (N, 3)
    :param downsampling_voxelsize:  下采样体素大小 (float),用于减少点云密度.默认为 2
    :param nb_points:  邻域点数 (int),用于统计离群点.默认为 7
    :param radius:  搜索半径 (float),用于确定邻域.默认为 3
    :param estimate_normals:  是否估计法线 (bool),如果为 True,则会估计点云的法线.默认为 False
    :param toggledebug:  是否开启调试模式 (bool),如果为 True,则会显示移除离群点过程中的中间结果.默认为 False
    :return: tuple: 包含清理后的点云和法线的元组,格式为 [pcd_nparray, nrmls_nparray] 或 [pcd_nparray]
             - pcd_nparray (numpy.ndarray):  清理后的点云,形状为 (M, 3)
             - nrmls_nparray (numpy.ndarray, optional):  点云的法线,形状为 (M, 3),仅当 `estimate_normals` 为 True 时返回

    author: weiwei
    date: 20191229
    """
    src_o3d = pda.nparray_to_o3dpcd(src_nparray, estimate_normals=estimate_normals)
    cl = _removeoutlier_o3d(src_o3d, downsampling_voxelsize, nb_points, radius, toggledebug)
    return pda.o3dpcd_to_nparray(cl, return_normals=estimate_normals)


def _removeoutlier_o3d(src_o3d, downsampling_voxelsize=2, nb_points=7, radius=3, toggledebug=False):
    """
    通过统计方法下采样并去除离群点

    :param src_o3d: 输入的点云(Open3D 格式)
    :param downsampling_voxelsize: 下采样的体素大小
    :param nb_points: 统计去除离群点时考虑的邻域点数
    :param radius: 离群点检测的半径
    :param toggledebug: 是否开启调试模式
    :return: 清理后的 Open3D 点云

    author: weiwei
    date: 20200316
    """
    if downsampling_voxelsize is None:
        src_o3d_down = src_o3d
    else:
        src_o3d_down = src_o3d.voxel_down_sample(downsampling_voxelsize)
    cl, _ = src_o3d_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    if toggledebug:
        src_o3d_down.paint_uniform_color([1, 0, 0])  # 将下采样的点云涂成红色
        src_o3d.paint_uniform_color([0, 1, 0])  # 将原始点云涂成绿色
        cl.paint_uniform_color([0, 0, 1])  # 将清理后的点云涂成蓝色
        o3d.visualization.draw_geometries([src_o3d_down, src_o3d, cl])
    return cl


def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """
    使用 ICP 算法进行点云配准

    :param src: 源点云
    :param tgt: 目标点云
    :param inithomomat: 初始变换矩阵
    :param maxcorrdist: 最大对应距离
    :param toggledebug: 是否开启调试模式
    :return: [匹配点的 RMSE, 匹配区域的大小, 变换矩阵]

    author: weiwei
    date: 20191229
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,  # 收敛条件:如果适应度 fitnesss 小于此值则收敛
                                                                 relative_rmse=1e-6,  # 收敛条件:如果 RMSE 小于此值则收敛
                                                                 max_iteration=2000)  # 最大迭代次数
    result_icp = o3d.pipelines.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    if toggledebug:
        __draw_registration_result(src, tgt, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.fitness, result_icp.transformation]


def cluster_pcd(pcd_nparray, pcd_nparray_nrmls=None):
    """
    使用 DBSCAN 方法对点云进行聚类

    :param pcd_nparray: 输入的点云数据(NumPy 数组格式)
    :param pcd_nparray_nrmls: 输入的法线数据(可选)
    :return: [聚类后的点云列表, 聚类后的法线列表]

    author: weiwei
    date: 20200316
    """
    db = skc.DBSCAN(eps=10, min_samples=50, n_jobs=-1).fit(pcd_nparray) # 使用 DBSCAN 进行聚类
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 计算聚类数量
    unique_labels = set(labels)
    pcd_nparray_list = []
    pcdnrmls_nparray_list = []
    if pcd_nparray_nrmls is None:
        pcd_nparray_nrmls = np.array([[0, 0, 1]] * pcd_nparray.shape[0])  # 默认法线为 (0, 0, 1)
    for k in unique_labels:
        if k == -1:  # 跳过噪声点
            continue
        else:
            class_member_mask = (labels == k)
            temppartialpcd = pcd_nparray[class_member_mask & core_samples_mask] # 获取聚类点
            pcd_nparray_list.append(temppartialpcd)
            temppartialpcdnrmls = pcd_nparray_nrmls[class_member_mask & core_samples_mask] # 获取对应法线
            pcdnrmls_nparray_list.append(temppartialpcdnrmls)
    return [pcd_nparray_list, pcdnrmls_nparray_list]


def merge_pcd(pnppcd1, pnppcd2, rotmat2, posmat2):
    """

    merge nppcd2 and nppcd1 by rotating and moving nppcd2 using rotmat2 and posmat2
    :param pnppcd1:
    :param pnppcd2:
    :param rotmat2:
    :param posmat2:
    :return:

    author: weiwei
    date: 20200221
    """
    transformednppcd2 = np.dot(rotmat2, pnppcd2.T).T + posmat2
    mergednppcd = np.zeros((len(transformednppcd2) + len(pnppcd1), 3))
    mergednppcd[:len(pnppcd1), :] = pnppcd1
    mergednppcd[len(pnppcd1):, :] = transformednppcd2
    return mergednppcd


def reconstruct_surfaces_bp(nppcd, nppcdnrmls=None, radii=[5], doseparation=True):
    """

    :param nppcd:
    :param radii:
    :param doseparation: separate the reconstructed meshes or not when they are disconnected
    :return:
    """
    if doseparation:
        # db = skc.DBSCAN(eps=10, min_samples=50).fit(nppcd)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # unique_labels = set(labels)
        # nppcdlist = []
        # nppcdnrmlslist = []
        # if nppcdnrmls is None:
        #     nppcdnrmls = np.array([[0,0,1]]*nppcd.shape[0])
        # else:
        #     nppcdnrmls = nppcdnrmls
        # for k in unique_labels:
        #     if k == -1:
        #         continue
        #     else:
        #         class_member_mask = (labels == k)
        #         temppartialpcd = nppcd[class_member_mask & core_samples_mask]
        #         nppcdlist.append(temppartialpcd)
        #         temppartialpcdnrmls = nppcdnrmls[class_member_mask & core_samples_mask]
        #         nppcdnrmlslist.append(temppartialpcdnrmls)
        nppcdlist, nppcdnrmlslist = cluster_pcd(nppcd, nppcdnrmls)

        tmmeshlist = []
        for i, thisnppcd in enumerate(nppcdlist):
            o3dpcd = pda.nparray_to_o3dpcd(thisnppcd, nppcdnrmlslist[i])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd,
                                                                                   o3d.utility.DoubleVector(radii))
            # meshes.compute_vertex_normals()
            mesh.compute_triangle_normals()
            tmmesh = pda.o3dmesh_to_trimesh(mesh)
            tmmeshlist.append(tmmesh)
        return tmmeshlist, nppcdlist
    else:
        if nppcdnrmls is None:
            npnrmls = np.array([[0, 0, 1]] * nppcd.shape[0])
        else:
            npnrmls = nppcdnrmls
        o3dpcd = pda.nparray_to_o3dpcd(nppcd, npnrmls)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd, o3d.utility.DoubleVector(radii))
        # meshes.compute_vertex_normals()
        mesh.compute_triangle_normals()
        tmmesh = pda.o3dmesh_to_trimesh(mesh)
        return tmmesh


def get_obb(pnppcd):
    """
    :param pnppcd:
    :return: [center_3x1nparray, corners_8x3nparray]
    author:
    date:
    """
    # TODO get the objects oriented bounding box of a point cloud using PoindCloud.get_oriented_bounding_box() and OrientedBoundinBox
    pass
