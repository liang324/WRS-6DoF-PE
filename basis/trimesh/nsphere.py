# nsphere.py
# 拟合和最小化nsphere的函数:
# 圆、球体、超球体等


import numpy as np
from . import util
from . import convex
from .constants import log, tol

try:
    # scipy 是一个可选依赖项
    from scipy import spatial
    from scipy.optimize import leastsq
except BaseException as E:
    # 当有人尝试使用它时抛出异常
    from . import exceptions

    leastsq = exceptions.closure(E)
    spatial = exceptions.ExceptionModule(E)

try:
    import psutil

    def _MAX_MEMORY():
        # 如果我们有 psutil,则在调用时检查实际的可用内存
        return psutil.virtual_memory().free / 2.0
except BaseException:
    def _MAX_MEMORY():
        # 使用硬编码的最佳猜测估计
        return 1e9


def minimum_nsphere(obj):
    """
    计算网格或一组点的最小 n-球

    使用的原理是最小 n-球的中心将位于最远点 Voronoi 图的一个顶点上,
    这需要 n*log(n) 的时间复杂度,但由于使用了 scipy/qhull 实现的凸包和 Voronoi 图,应该会非常快

    :param obj: (n, d) float 或 trimesh.Trimesh 要查找最小包围 n-球的点或网格
    :return:      center : (d,) float 拟合 n-球的中心
      radius : float 拟合 n-球的半径
    """
    # 将输入点或网格减少到凸包的顶点
    # 由于我们正在计算最远的站点voronoi图,这减少了
    # 输入复杂度本质上是相同的,并返回相同的值
    points = convex.hull_points(obj)

    # 我们将网格缩放到单位立方体
    # 这用于将qhull_options ` QbB `传递给Voronoi,然而这在某处有一个bug
    # 为了避免这种情况,我们在这个函数中自己扩展到一个单位立方体
    points_origin = points.min(axis=0)
    points_scale = points.ptp(axis=0).min()
    points = (points - points_origin) / points_scale

    # 如果所有的点都在一个n球面上,那就是voronoi图
    # 方法将失败,因此我们先检查最小二乘拟合
    # 计算voronoi图很麻烦
    fit_C, fit_R, fit_E = fit_nsphere(points)
    # 返回半径和中心到全局范围
    fit_R = (((points - fit_C) ** 2).sum(axis=1).max() ** .5) * points_scale
    fit_C = (fit_C * points_scale) + points_origin
    if fit_E < 1e-6:
        log.debug('点在一个 n-球上,返回拟合结果')
        return fit_C, fit_R

    # 计算最远的站点voronoi图
    # 如果所有点都在表面上,则失败
    # n球面但愿最小二乘法能解决这些问题
    # , qhull_options='QbB Pp')
    voronoi = spatial.Voronoi(points, furthest_site=True)

    # 找出每个voronoi顶点的最大半径^2点
    # 这是非常耗时的最坏情况,但我们已经采取了
    # 凸包来减少n
    # 我们正在比较半径的平方,然后生根一次
    try:
        # cdist比循环或平铺方法快得多
        # 尽管它创建了一个非常大的中间数组
        # 首先,估计一个数量级的内存大小
        # 一个float64类型的数组项是8字节,加上开销
        memory_estimate = len(voronoi.vertices) * len(points) * 9
        if memory_estimate > _MAX_MEMORY():
            raise MemoryError
        radii_2 = spatial.distance.cdist(voronoi.vertices, points, metric='sqeuclidean').max(axis=1)
    except MemoryError:
        # 记录MemoryError
        log.warning('MemoryError: 回退到较慢的检查！')
        # 回退到可能非常慢的列表推导
        radii_2 = np.array([((points - v) ** 2).sum(axis=1).max() for v in voronoi.vertices])
    # 我们想要最小的球体,因此取半径的最小值
    radii_idx = radii_2.argmin()
    # 返回 Voronoi 半径和中心到全局缩放
    radius_v = np.sqrt(radii_2[radii_idx]) * points_scale
    center_v = (voronoi.vertices[radii_idx] * points_scale) + points_origin
    if radius_v > fit_R:
        return fit_C, fit_R
    return center_v, radius_v


def fit_nsphere(points, prior=None):
    """
    使用最小二乘法拟合一个 n 维球体到一组点

    :param points: (n, d) float,空间中的点
    :param prior: (d,) float,n 维球体中心的最佳猜测
    :return:
        - center: (d,) float,中心的位置
        - radius: float,圆的平均半径
        - error: float,偏离平均半径的峰值
    """
    # 确保点是 numpy 数组
    points = np.asanyarray(points, dtype=np.float64)
    # 创建全为一的数组,以便可以使用点积而不是较慢的求和
    ones = np.ones(points.shape[1])

    def residuals(center):
        # 使用点积进行轴求和,这个函数被调用很多次,所以值得优化
        radii_sq = np.dot((points - center) ** 2, ones)
        # 残差是与平均值的差异,使用我们的求和平均值而不是 .mean() 因为它稍微快一点
        return radii_sq - (radii_sq.sum() / len(radii_sq))
    if prior is None:
        guess = points.mean(axis=0)
    else:
        guess = np.asanyarray(prior)
    center_result, return_code = leastsq(residuals, guess, xtol=1e-8)
    if not (return_code in [1, 2, 3, 4]):
        raise ValueError('最小二乘拟合失败！')
    radii = util.row_norm(points - center_result)
    radius = radii.mean()
    error = radii.ptp()
    return center_result, radius, error


def is_nsphere(points):
    """
    检查一组点是否为 n-球
    
    :param points: (n, dimension) float 空间中的点
    :return: check : bool 如果输入点在 n-球上,则为 True
    """
    center, radius, error = fit_nsphere(points)
    check = error < tol.merge
    return check
