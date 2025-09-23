import numpy as np
import basis.trimesh as trm
import basis.robot_math as rm


def scale(obj, scale_ratio):
    """
    缩放三维模型或文件路径指定的模型

    :param obj: Trimesh 对象或文件路径
    :param scale_ratio: float,等比例缩放所有轴
    :return: 缩放后的 Trimesh 对象

    author: weiwei
    date: 20201116
    """
    if isinstance(obj, trm.Trimesh):
        # 如果输入是 Trimesh 对象,复制并缩放
        tmpmesh = obj.copy()
        tmpmesh.apply_scale(scale_ratio)
        return tmpmesh
    elif isinstance(obj, str):
        originalmesh = trm.load(obj)
        # 如果输入是文件路径,加载模型后复制并缩放
        tmpmesh = originalmesh.copy()
        tmpmesh.apply_scale(scale_ratio)
        return tmpmesh


def scale_and_save(obj, scale_ratio, savename):
    """
    缩放三维模型并保存到指定路径

    :param obj: Trimesh 对象或文件路径
    :param scale_ratio: float,等比例缩放所有轴
    :param savename: 保存路径和文件名

    :return:
    author: weiwei
    date: 20201116
    """
    # 缩放模型
    tmptrimesh = scale(obj, scale_ratio)
    # 导出缩放后的模型
    tmptrimesh.export(savename)


def convert_to_stl(obj, savename, scale_ratio=1, pos=np.zeros(3), rotmat=np.eye(3)):
    """
    转换三维模型为 STL 格式,并应用缩放、平移和旋转

    :param obj: Trimesh 对象或文件路径
    :param savename: 保存路径和文件名
    :param scale_ratio: 缩放比例,默认为 1
    :param pos: 平移向量,默认为零向量
    :param rotmat: 旋转矩阵,默认为单位矩阵
    :param obj: trimesh or file path
    :param savename:
    :return:

    author: weiwei
    date: 20201207
    """
    # 加载模型
    trimesh = trm.load(obj)
    # 如果缩放比例不是列表,转换为列表
    if type(scale_ratio) is not list:
        scale_ratio = [scale_ratio, scale_ratio, scale_ratio]
    # 缩放模型
    tmptrimesh = scale(trimesh, scale_ratio)
    # 生成齐次变换矩阵
    homomat = rm.homomat_from_posrot(pos, rotmat)
    # 应用变换
    tmptrimesh.apply_transform(homomat)
    # 导出为 STL 格式
    tmptrimesh.export(savename)


if __name__ == '__main__':
    # 为避免误执行,以下内容被注释掉
    pass
    # root = "./objects/"
    # for subdir, dirs, files in os.walk(root):
    #     for file in files:
    #         print(root+file)
    #         scale_and_save(root+file, .001, file)
    # scale_and_save("./objects/block.meshes", .001, "block.meshes")
    # scale_and_save("./objects/bowlblock.meshes", .001, "bowlblock.meshes")
    convert_to_stl("base.dae", "base.meshes")
