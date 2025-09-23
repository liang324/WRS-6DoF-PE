import numpy as np
import open3d as o3d
import copy
import sklearn.cluster as skc
import basis.trimesh as trimesh


# abbreviations
# pnp panda nodepath
# o3d open3d
# pnppcd - a point cloud in the panda nodepath format


def nparray_to_o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    """
    将 n-by-3 的 numpy 数组转换为 Open3D 点云对象

    :param nx3nparray_pnts: (n,3) numpy 数组,表示点云的坐标
    :param nx3nparray_nrmls: (n,3) numpy 数组,表示点云的法向量
    :param estimate_normals: 如果 nx3nparray_nrmls 为 None,则根据此参数决定是否估计法向量
    :return: Open3D 点云对象

    author: ruishuang, weiwei
    date: 20191210
    """
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


def o3dpcd_to_parray(o3dpcd, return_normals=False):
    """
    将 Open3D 点云对象转换为 numpy 数组

    :param o3dpcd: Open3D 点云对象
    :param return_normals: 是否返回法向量
    :return: numpy 数组,包含点云坐标和可选的法向量

    author:  weiwei
    date: 20191229, 20200316
    """
    if return_normals:
        if o3dpcd.has_normals():
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
        else:
            o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
    else:
        return np.asarray(o3dpcd.points)


def o3dmesh_to_trimesh(o3dmesh):
    """
    将 Open3D 网格对象转换为 trimesh 网格对象

    :param o3dmesh: Open3D 网格对象
    :return: trimesh 网格对象

    author: weiwei
    date: 20191210
    """
    vertices = np.asarray(o3dmesh.vertices)
    faces = np.asarray(o3dmesh.triangles)
    face_normals = np.asarray(o3dmesh.triangle_normals)
    cvterd_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)
    return cvterd_trimesh


def crop_o3dpcd(o3dpcd, xrng, yrng, zrng):
    """
    裁剪 Open3D 点云对象

    :param o3dpcd: Open3D 点云对象
    :param xrng, yrng, zrng: [min, max],裁剪范围
    :return: 裁剪后的 Open3D 点云对象

    author: weiwei
    date: 20191210
    """
    o3dpcdarray = np.asarray(o3dpcd.points)
    xmask = np.logical_and(o3dpcdarray[:, 0] > xrng[0], o3dpcdarray[:, 0] < xrng[1])
    ymask = np.logical_and(o3dpcdarray[:, 1] > yrng[0], o3dpcdarray[:, 1] < yrng[1])
    zmask = np.logical_and(o3dpcdarray[:, 2] > zrng[0], o3dpcdarray[:, 2] < zrng[1])
    mask = xmask * ymask * zmask
    return nparray_to_o3dpcd(o3dpcdarray[mask])


def crop_nx3_nparray(nx3nparray, xrng, yrng, zrng):
    """
    裁剪 n-by-3 的 numpy 数组

    :param nx3nparray: n-by-3 numpy 数组
    :param xrng, yrng, zrng: [min, max],裁剪范围
    :return: 裁剪后的 numpy 数组

    author: weiwei
    date: 20191210
    """
    xmask = np.logical_and(nx3nparray[:, 0] > xrng[0], nx3nparray[:, 0] < xrng[1])
    ymask = np.logical_and(nx3nparray[:, 1] > yrng[0], nx3nparray[:, 1] < yrng[1])
    zmask = np.logical_and(nx3nparray[:, 2] > zrng[0], nx3nparray[:, 2] < zrng[1])
    mask = xmask * ymask * zmask
    return nx3nparray[mask]
