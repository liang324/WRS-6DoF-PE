import numpy as np
from .util import zero_pad, format_json
from .grouping import group_rows
from .constants import log, _log_time

_MIN_BIN_COUNT = 20
_TOL_FREQ = 1e-3


def rotationally_invariant_identifier(mesh, length=6, as_json=False, json_digits=None):
    '''
    给定一个输入网格,返回一个具有以下属性的向量或字符串:
    * 对网格的旋转不变
    * 对表面不同的细分具有鲁棒性
    * 相似但不相同的网格返回的值在欧几里得距离上接近

    通过计算半径的面积加权分布(从质心)来实现

    :param mesh: Trimesh 对象
    :param length: 计算标识符的项数
    :param as_json: 是否将标识符作为 JSON 返回(与 1D 浮点数组相对)

    :return identifier: 如果 not as_json,则返回 (length) 浮点数组的唯一标识符
                        否则,返回相同的内容,但序列化为 JSON
    '''

    frequency_count = int(length - 2)
    # 计算网格的质量属性,这是通过表面积分来找到网格的体积中心
    mass_properties = mesh.mass_properties(skip_inertia=True)
    vertex_radii = np.sum((mesh.vertices.view(np.ndarray) - mesh.center_mass) ** 2, axis=1) ** .5

    # 由于我们将计算半径的形状分布,我们需要确保有足够的值来填充每个 bin 的多个样本
    bin_count = int(np.min([256,
                            mesh.vertices.shape[0] * 0.2,
                            mesh.faces.shape[0] * 0.2]))

    # 如果任何频率检查失败,我们将使用这个零长度向量作为标识符的格式化信息
    freq_formatted = np.zeros(frequency_count)

    if bin_count > _MIN_BIN_COUNT:
        face_area = mesh.area_faces
        face_radii = vertex_radii[mesh.faces].reshape(-1)
        area_weight = np.tile((face_area.reshape((-1, 1)) * (1.0 / 3.0)), (1, 3)).reshape(-1)

        if face_radii.std() > 1e-3:
            freq_formatted = fft_freq_histogram(face_radii,
                                                bin_count=bin_count,
                                                frequency_count=frequency_count,
                                                weight=area_weight)

    # 使用体积(从表面积分)、表面积和最高频率
    identifier = np.hstack((mass_properties['volume'],
                            mass_properties['surface_area'],
                            freq_formatted))
    if as_json:
        # 返回 JSON 字符串而不是数组
        return format_json(identifier)
    return identifier


def fft_freq_histogram(data, bin_count, frequency_count=4, weight=None):
    data = np.reshape(data, -1)
    if weight is None:
        weight = np.ones(len(data))

    hist, bin_edges = np.histogram(data, weights=weight, bins=bin_count)
    # 我们计算半径分布的 FFT
    fft = np.abs(np.fft.fft(hist))
    # 幅度取决于我们的加权是否良好频率在更多情况下应该更稳定
    freq = np.fft.fftfreq(data.size, d=(bin_edges[1] - bin_edges[0])) + bin_edges[0]

    # 现在我们必须选择最高的 FREQ_COUNT 频率
    # 如果有一堆频率,其分量在幅度上非常接近,
    # 仅选择最高的 FREQ_COUNT 可能是非确定性的
    # 因此我们选择具有可区分幅度的最高频率
    # 如果这意味着可用的值较少,我们将进行零填充
    fft_top = fft.argsort()[-(frequency_count + 1):]
    fft_ok = np.diff(fft[fft_top]) > _TOL_FREQ
    if fft_ok.any():
        fft_start = np.nonzero(fft_ok)[0][0] + 1
        fft_top = fft_top[fft_start:]
        freq_final = np.sort(freq[fft_top])
    else:
        freq_final = []

    freq_formatted = zero_pad(freq_final, frequency_count)
    return freq_formatted


@_log_time
def merge_duplicates(meshes):
    '''
    给定一个网格列表,找到重复的网格并合并它们
    :param meshes: (n) 网格列表
    :return merged: (m) 网格列表,其中 (m <= n)
    '''
    # 以便我们可以使用高级索引
    meshes = np.array(meshes)
    # 默认情况下,标识符是一个具有 6 个元素的 1D 浮点数组
    hashes = [i.identifier for i in meshes]
    groups = group_rows(hashes, digits=1)
    merged = [None] * len(groups)
    for i, group in enumerate(groups):
        quantity = 0
        metadata = {}
        for mesh in meshes[group]:
            # 如果元数据存在,不要删除它
            if 'quantity' in mesh.metadata:
                quantity += mesh.metadata['quantity']
            else:
                quantity += 1
            metadata.update(mesh.metadata)

        metadata['quantity'] = int(quantity)
        metadata['original_index'] = group

        merged[i] = meshes[group[0]]
        merged[i].metadata = metadata
    log.info('merge_duplicates 将部件数量从 %d 减少到 %d',
             len(meshes),
             len(merged))
    return np.array(merged)
