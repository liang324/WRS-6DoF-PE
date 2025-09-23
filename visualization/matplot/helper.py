import numpy as np
import matplotlib.pyplot as plt

PIXEL_TO_INCH = 0.0104166667


def list_to_plt_xy(data_list):
    """
    将一维数据列表转换为 matplotlib 可用的 x 和 y 数据

    :param data_list: 一维数据列表
    :return: x 数据(索引范围)和 y 数据(原始数据列表)
    """
    return range(len(data_list)), data_list


def twodlist_to_plt_xys(data_2dlist):
    """
    将二维数据列表转换为 matplotlib 可用的 x 和 y 数据

    :param data_2dlist: 二维数据列表
    :return: x 数据(索引范围)和 y 数据(转换为 numpy 数组)
    """
    data_array = np.array(data_2dlist)
    return range(data_array.shape[0]), data_array
