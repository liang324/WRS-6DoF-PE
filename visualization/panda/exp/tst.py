import os
import open3d as o3d


def get_objpcd_partial_o3d(objcm, path='./', f_name='', resolusion=(1280, 720), ext_name='.pcd', ):
    """
    从给定的三维模型生成部分点云并保存到指定路径

    :param objcm: 包含三维模型信息的对象,必须具有 objtrm 属性,其中包含顶点和面信息
    :param path: 保存点云文件的路径,默认为当前目录 ('./')
    :param f_name: 保存文件的基础名称,默认为空字符串 ('')
    :param resolusion: 窗口的分辨率,默认为 (1280, 720)
    :param ext_name: 保存文件的扩展名,默认为 '.pcd'
    :return: 返回一个 Open3D 点云对象,包含捕获的部分点云数据
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    o3dmesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                        triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))
    o3dmesh.compute_vertex_normals()
    # o3dmesh.rotate(rot, center=rot_center)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f_name + f'_partial_org{ext_name}'), do_render=False,
                                  convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f_name + f'_partial_org{ext_name}'))

    vis.destroy_window()

    return o3dpcd


if __name__ == "__main__":
    import modeling.collision_model as cm
    import numpy as np
    from huri.core.common_import import wd, gm

    obj = cm.CollisionModel("bunnysim.stl")
    pcdo3d = get_objpcd_partial_o3d(obj)
    pcdnp = np.asarray(pcdo3d.points)
    base = wd.World(cam_pos=[0, 0, .3], lookat_pos=[0, 0, .0])
    gm.gen_pointcloud(pcdnp).attach_to(base)
    obj.attach_to(base)
    base.run()
