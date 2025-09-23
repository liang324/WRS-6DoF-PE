# inertia.py 处理惯性张量 inertia tensor 的函数结果根据已知的几何形状进行验证和检查内部一致性

import numpy as np
from . import util


def cylinder_inertia(mass, radius, height, transform=None):
    """
    返回圆柱体的惯性张量

    :param mass: float,圆柱体的质量
    :param radius: float,圆柱体的半径
    :param height: float,圆柱体的高度
    :param transform: (4, 4) float,圆柱体的变换矩阵
    :return: inertia: (3, 3) float,惯性张量
    """
    h2, r2 = height ** 2, radius ** 2
    diagonal = np.array([((mass * h2) / 12) + ((mass * r2) / 4),
                         ((mass * h2) / 12) + ((mass * r2) / 4),
                         (mass * r2) / 2])
    inertia = diagonal * np.eye(3)
    if transform is not None:
        inertia = transform_inertia(transform, inertia)
    return inertia


def sphere_inertia(mass, radius):
    """
    返回球体的惯性张量

    :param mass: float,球体的质量
    :param radius: float,球体的半径
    :return: inertia: (3, 3) float,惯性张量
    """
    inertia = (2.0 / 5.0) * (radius ** 2) * mass * np.eye(3)
    return inertia


def principal_axis(inertia):
    """
    从惯性张量中找到惯性主成分和主轴

    :param inertia: (3, 3) float,惯性张量
    :return: components: (3,) float,惯性主成分
             vectors: (3, 3) float,指向惯性主轴的行向量
    """
    inertia = np.asanyarray(inertia, dtype=np.float64)
    if inertia.shape != (3, 3):
        raise ValueError('惯性张量必须是 (3, 3)!')

    # 可以使用以下任何一种方法来计算:
    # np.linalg.svd, np.linalg.eig, np.linalg.eigh
    # 惯性矩是方对称矩阵,在测试中,eigh 具有最佳的数值精度
    components, vectors = np.linalg.eigh(inertia)
    # eigh 返回列向量,将其更改为行向量
    vectors = vectors.T
    return components, vectors


def transform_inertia(transform, inertia_tensor):
    """
    将惯性张量转换到新的坐标系

    :param transform: (3, 3) 或 (4, 4) float,变换矩阵
    :param inertia_tensor: (3, 3) float,惯性张量
    :return: transformed: (3, 3) float,新坐标系下的惯性张量
    """
    # 检查输入并提取旋转
    transform = np.asanyarray(transform, dtype=np.float64)
    if transform.shape == (4, 4):
        rotation = transform[:3, :3]
    elif transform.shape == (3, 3):
        rotation = transform
    else:
        raise ValueError('变换矩阵必须是 (3, 3) or (4, 4)!')
    inertia_tensor = np.asanyarray(inertia_tensor, dtype=np.float64)
    if inertia_tensor.shape != (3, 3):
        raise ValueError('inertia_tensor 惯性张量必须是 (3, 3)!')
    transformed = util.multi_dot([rotation,
                                  inertia_tensor,
                                  rotation.T])
    return transformed


def radial_symmetry(mesh):
    """
    检查网格是否具有径向对称性

    :param mesh: Trimesh 对象
    :return: symmetry: None 或 str
             None         无旋转对称性
             'radial'     围绕轴对称
             'spherical'  围绕点对称
             axis: None 或 (3,) float
             旋转轴或点
             section: None 或 (3, 2) float
             如果是径向对称性,提供向量以获得截面
    """
    # 快捷方式以避免重复输入和命中缓存
    scalar = mesh.principal_inertia_components
    vector = mesh.principal_inertia_vectors
    # 主成分的排序顺序
    order = scalar.argsort()
    # 我们正在检查几何体是否具有径向对称性
    # 如果两个主惯性成分(PCI)相等,则它是一个旋转的二维轮廓
    # 如果三个主惯性成分(所有成分)相等,则它是一个球体
    # 因此,我们对排序后的主惯性成分进行差分,按最大主惯性成分的比例进行缩放,
    # 然后缩放到我们关心的容差
    # 如果容差为 1e-3,则意味着两个成分是相同的,如果它们在最大主惯性成分的 0.1% 以内.
    diff = np.abs(np.diff(scalar[order]))
    diff /= np.abs(scalar).max()
    # 在容差范围内为零的差分
    diff_zero = (diff / 1e-3).astype(int) == 0
    if diff_zero.all():
        # 所有三个主惯性成分相同的情况
        # 这意味着几何体围绕一个点对称
        # 例如球体、二十面体等
        axis = vector[0]
        section = vector[1:]
        return 'spherical', axis, section

    elif diff_zero.any():
        # 2/3 PCI的情况相同
        # 这意味着几何图形是关于轴对称的
        # 可能是旋转的2D剖面图
        # 我们知道只有1/2的diff值为真
        # 如果第一个diff为0,则表示我们取第一个元素
        # 在有序PCI中,我们将有一个非旋转轴
        # 如果第二个diff为0,则取的最后一个元素
        # section轴的有序PCI
        # 如果我们想要旋转轴,我们只需要将[0,-1]切换到
        # (1,0)
        # 由于两个向量相同,我们知道中间的一个是其中之一
        section_index = order[np.array([[0, 1], [1, -1]])[diff_zero]].flatten()
        section = vector[section_index]
        # 我们知道旋转轴是唯一的值,并且是排序值中的第一个或最后一个
        axis_index = order[np.array([-1, 0])[diff_zero]][0]
        axis = vector[axis_index]
        return 'radial', axis, section
    return None, None, None
