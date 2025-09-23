import numpy as np
from collections import deque
from .arc import arc_center, fit_circle, angles_to_threepoint
from .entities import Arc, Line, BSpline
from ..util import unitize, diagonal_dot
from ..constants import log
from ..constants import tol_path as tol


def fit_circle_check(points, prior=None, scale=1.0, verbose=False):
    '''
    拟合一个圆,并在以下情况下拒绝拟合:
    - 半径大于 tol.radius_min*scale 或 tol.radius_max*scale
    - 任何线段跨度超过 tol.seg_angle
    - 任何线段长度超过 tol.seg_frac*scale
    - 拟合偏差超过 tol.radius_frac*radius
    - 末端线段偏离切线超过 tol.tangent

    :param points: (n, d) 表示路径的一组点
    :param prior: (center, radius) 元组,表示最佳猜测,或未知时为 None
    :param scale: float,表示点集的整体比例
    :param verbose: boolean,如果为 True,则输出 log.debug 消息以说明拟合拒绝的原因.
                    可能会生成成千上万条消息,因此仅建议在手动调试时使用
    :return: 如果拟合可接受,返回 (center, radius) 元组；否则返回 None
    '''
    # 拟合一个圆,并进行容差检查
    if len(points) < 3: return None
    # 对点进行最小二乘拟合
    C, R, r_deviation = fit_circle(points, prior=prior)

    # 检查半径是否在允许的最小值和最大值之间
    if not tol.radius_min < (R / scale) < tol.radius_max:
        if verbose:
            log.debug('圆拟合错误: 半径R %f', R / scale)
        return None

    # 检查点的半径误差
    r_error = r_deviation / R
    if r_error > tol.radius_frac:
        if verbose:
            log.debug('圆拟合错误: 拟合 %s', str(r_error))
        return None

    vectors = np.diff(points, axis=0)
    segment = np.linalg.norm(vectors, axis=1)

    # 检查线段长度是否超过绘图比例的分数
    scaled = segment / scale
    # 近似角度(弧度),线段是线性长度而不是弧长,但这很接近并避免了余弦
    angle = segment / R

    if (angle > tol.seg_angle).any():
        if verbose:
            log.debug('圆拟合错误: 角度 %s', str(angle))
        return None

    if (scaled > tol.seg_frac).any():
        if verbose:
            log.debug('圆拟合错误: 线段 %s', str(scaled))
        return None

    # 确保末端的线段实际上与候选圆拟合相切
    mid_pt = points[[0, -2]] + (vectors[[0, -1]] * .5)
    radial = unitize(mid_pt - C)
    ends = unitize(vectors[[0, -1]])
    tangent = np.abs(np.arccos(diagonal_dot(radial, ends)))
    tangent = np.abs(tangent - np.pi / 2).max()
    if tangent > tol.tangent:
        if verbose:
            log.debug('圆拟合错误: 切线 %f', np.degrees(tangent))
        return None
    return (C, R)


def is_circle(points, scale, verbose=True):
    '''
    给定一组点,快速确定它们是否表示一个圆

    :param points: (n, d) 表示路径的一组点
    :param scale: float,表示点集的整体比例
    :param verbose: boolean,如果为 True,则输出 log.debug 消息以说明拟合拒绝的原因.
    :return: 如果是圆,返回三个控制点；否则返回 None
    '''
    # 确保输入是一个 numpy 数组
    points = np.asanyarray(points)
    scale = float(scale)

    # 只有当第一个和最后一个点相同时(即路径是闭合的),才能是一个圆
    if np.linalg.norm(points[0] - points[-1]) > tol.merge:
        return None

    box = points.ptp(axis=0)
    # 检查长宽比,如果路径不是圆形,则提前退出
    aspect = np.divide(*box)
    if np.abs(aspect - 1.0) > tol.aspect_frac:
        return None

    # 拟合一个圆并进行容差检查
    CR = fit_circle_check(points, scale=scale)
    if CR is None:
        return None

    # 将圆作为三个控制点返回
    control = angles_to_threepoint([0, np.pi * .5], *CR)
    return control


def arc_march(points, scale):
    '''
    使用最小二乘拟合将路径分割为直线和弧段

    :param points: (n, d) 表示路径的一组点
    :param scale: 缩放比例,用于确定弧的跨度
    :return: arcs (b) 可以用弧替换的点索引序列
    '''

    def finalize_arc(points_id):
        # 对当前包含的点进行最终检查,如果通过则将它们添加到弧列表中
        points_id = np.array(points_id)
        try:
            center_info = arc_center(points)
            C, R, N, A = (center_info['center'],
                          center_info['radius'],
                          center_info['normal'],
                          center_info['span'])
        except ValueError:
            return

        span = scale * (A / R)
        if span > 1.5:
            arcs.append(points_id)
        else:
            log.debug('弧未通过跨度测试:  %f', span)

    points = np.array(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge
    count = len(points)
    scale = float(scale)
    # if scale is None:
    #    scale = np.ptp(points, axis=0).max()

    arcs = deque()
    current = deque()
    prior = None

    # 确定遍历点的次数
    # 如果点是闭合的,则最多遍历两次
    attempts = count + count * int(closed)

    for index in range(attempts):
        # 检查是否已经遍历了这些点
        looped = index >= count
        # 确保索引在范围内
        i = index % count

        # 如果已经遍历过,说明这些点是闭合的
        # 如果它们是闭合的,说明 points[0] == points[-1]
        # 因此,在第二次遍历时,我们要跳过索引 0 以避免处理重复点
        if looped and i == 0: continue
        # 将索引添加到当前候选集中
        current.append(i)
        # 如果当前候选集中的点少于三个,它们不能构成弧
        if (len(current) < 3): continue

        # 拟合一个圆,如果不符合公差,则拒绝拟合,如果拟合被拒绝,fit_circle_check 将返回 None
        checked = fit_circle_check(points[current],
                                   prior=prior,
                                   scale=scale,
                                   verbose=True)
        arc_ok = checked is not None

        # 由于我们要遍历两次点,在第二次遍历时,我们只想在弧拟合失败时停止
        ending = looped and (not arc_ok)
        ending = ending or (index >= attempts - 1)

        if ending and prior is not None:
            # 我们可能是因为拟合不好而停止
            # 或者我们可能只是用完了索引.
            # 如果是拟合不好,移除最后添加的点.
            if not arc_ok:
                current.pop()
            # 如果我们正在停止并且拟合可接受,则添加弧
            finalize_arc(current)
        elif arc_ok:
            # 如果我们没有结束并且拟合看起来不错,只需用拟合更新先前的值
            prior = checked
        elif prior is None:
            # 我们还没有看到一个可接受的拟合 所以从左边移除一个索引
            current.popleft()
        else:
            # 弧不合适,并且我们有一个拟合,所以移除
            # 最新的点然后添加弧
            current.pop()
            finalize_arc(current)
            # 重置候选集
            current = deque([i - 1, i])
            prior = None
        if ending: break
    if looped and len(arcs) > 0 and arcs[0][0] == 0:
        arcs.popleft()
    arcs = np.array(arcs)
    return arcs


def merge_colinear(points, scale=None):
    '''
    给定一组表示空间路径的点,合并共线的点

    :param points: (n, d) 一组点(其中 d 是维度)
    :param scale: float,绘图的缩放比例
    :return: merged (j, d) 合并共线和重复点后的点集,其中 (j < n)
    '''
    points = np.array(points)
    if scale is None:
        scale = np.ptp(points, axis=0).max()

    # 从一个点到下一个点的向量
    direction = np.diff(points, axis=0)
    # 方向向量的长度
    direction_norm = np.linalg.norm(direction, axis=1)
    # 确保点没有零长度
    direction_ok = direction_norm > tol.merge
    # 移除重复点
    points = np.vstack((points[0], points[1:][direction_ok]))
    direction = direction[direction_ok]
    direction_norm = direction_norm[direction_ok]
    # 将非零方向向量转换为单位向量
    direction /= direction_norm.reshape((-1, 1))
    # 找到后续方向向量之间的差异
    direction_diff = np.linalg.norm(np.diff(direction, axis=0), axis=1)
    # 向量之间的方向差异的大小乘以方向长度
    colinear = (direction_diff * direction_norm[1:]) < (tol.merge * scale)
    colinear_index = np.nonzero(colinear)[0]
    mask = np.ones(len(points), dtype=bool)
    # 由于我们使用了差分,需要偏移一个单位
    mask[colinear_index + 1] = False
    merged = points[mask]
    return merged


def resample_spline(points, smooth=.001, count=None):
    """
    使用样条曲线重新采样给定的点集

    :param points: 输入的点集
    :param smooth: 平滑因子,控制样条曲线的平滑程度
    :param count: 重新采样后的点数,如果为 None,则使用原始点数
    :return: 重新采样后的点集
    """
    from scipy.interpolate import splprep, splev
    if count is None:
        count = len(points)
    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

    # 使用样条曲线拟合点集
    tpl = splprep(points.T, s=smooth)[0]
    i = np.linspace(0.0, 1.0, count)
    resampled = np.column_stack(splev(i, tpl))

    if closed:
        # 如果点集是闭合的,调整首尾点以确保闭合
        shared = resampled[[0, -1]].mean(axis=0)
        resampled[0] = shared
        resampled[-1] = shared
    return resampled


def points_to_spline_entity(points, smooth=.0005, count=None):
    """
    将点集转换为样条实体

    :param points: 输入的点集
    :param smooth: 平滑因子,控制样条曲线的平滑程度
    :param count: 重新采样后的点数,如果为 None,则使用原始点数
    :return: 样条实体和控制点
    """
    from scipy.interpolate import splprep
    if count is None:
        count = len(points)
    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge
    # 使用样条曲线拟合点集
    knots, control, degree = splprep(points.T, s=smooth)[0]
    control = np.transpose(control)
    index = np.arange(len(control))
    if closed:
        # 如果点集是闭合的,调整首尾控制点以确保闭合
        control[0] = control[[0, -1]].mean(axis=0)
        control = control[:-1]
        index[-1] = index[0]
    entity = BSpline(points=index,
                     knots=knots,
                     closed=closed)
    return entity, control


def three_point(indices):
    """
    从索引列表中选择三个点:第一个、中间和最后一个

    :param indices: 输入的索引列表
    :return: 包含三个点的数组
    """
    result = [indices[0],
              indices[int(len(indices) / 2)],
              indices[-1]]
    return np.array(result)


def polygon_to_cleaned(polygon, scale):
    """
    清理多边形的外部轮廓,合并共线点

    :param polygon: 输入的多边形对象
    :param scale: 缩放比例,用于合并共线点
    :return: 清理后的点集
    """
    buffered = polygon.buffer(0.0)
    points = merge_colinear(buffered.exterior.coords)
    return points


def simplify_path(drawing):
    """
    简化仅包含直线段的路径,将其转换为包含拟合弧和圆的路径

    :param drawing: 包含路径信息的绘图对象
    """
    if any([i.__class__.__name__ != 'Line' for i in drawing.entities]):
        log.debug('路径包含非线性实体,跳过处理')
        return
    vertices = deque()
    entities = deque()
    for path_index in range(len(drawing.paths)):
        # 清理多边形并合并共线
        points = polygon_to_cleaned(drawing.polygons_closed[path_index], scale=drawing.scale)
        # 检查是否可以拟合为圆
        circle = is_circle(points, scale=drawing.scale)

        if circle is not None:
            # 如果是圆,添加为弧实体
            entities.append(Arc(points=np.arange(3) + len(vertices), closed=True))
            vertices.extend(circle)
        else:
            # 尝试将路径分割为弧段
            arc_idx = arc_march(points, scale=drawing.scale)
            if len(arc_idx) > 0:
                for arc in arc_idx:
                    entities.append(Arc(points=three_point(arc) + len(vertices), closed=False))
                # 填充弧段之间的直线段
                line_idx = infill_lines(arc_idx, len(points)) + len(vertices)
            else:
                # 如果没有弧段,直接连接起点和终点
                line_idx = pair_space(0, len(points) - 1) + len(vertices)
                line_idx = [np.mod(np.arange(len(points) + 1), len(points)) + len(vertices)]
            for line in line_idx:
                entities.append(Line(points=line))
            vertices.extend(points)
    # 清除缓存并更新绘图对象的顶点和实体
    drawing._cache.clear()
    drawing.vertices = np.array(vertices)
    drawing.entities = np.array(entities)


def pair_space(start, end):
    """
    生成从起始索引到结束索引的成对索引

    :param start: 起始索引
    :param end: 结束索引
    :return: 成对索引数组
    """
    if start == end: return []
    idx = np.arange(start, end + 1)
    idx = np.column_stack((idx, idx)).reshape(-1)[1:-1].reshape(-1, 2)
    return idx


def infill_lines(idxs, idx_max):
    """
    填充弧段之间的直线段

    :param idxs: 弧段索引数组
    :param idx_max: 最大索引值
    :return: 填充的直线段索引数组
    """
    if len(idxs) == 0:
        return np.array([])
    ends = np.array([i[[0, -1]] for i in idxs])
    ends = np.roll(ends.reshape(-1), -1).reshape(-1, 2)
    if np.greater(*ends[-1]):
        ends[-1][1] += idx_max
    infill = np.diff(ends, axis=1).reshape(-1) > 0
    aranges = ends[infill]
    if len(aranges) == 0:
        return np.array([])
    result = np.vstack([pair_space(*i) for i in aranges])
    result %= idx_max
    return result
