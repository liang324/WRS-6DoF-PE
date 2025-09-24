#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：WRS-6DoF-PE
@File    ：human_grasping.py
@IDE     ：PyCharm
@Author  ：Yixuan Su
@Date    ：2025/07/11 10:07
Description:

"""

import numpy as np


def is_robot_grabbable(matrix, row, col):
    """判断指定位置是否满足机器人抓取条件(上下或者左右没有试管)
    Args:
        matrix (np.ndarray): 试管架状态矩阵,1表示有试管,0表示无试管
        row (int): 试管在矩阵中的行索引
        col (int): 试管在矩阵中的列索引
    Returns:
        bool: 如果满足抓取条件,返回True；否则返回False
    """
    rows, cols = matrix.shape
    # 判断上下是否都没有试管
    vertical_clear = True
    if row > 0 and matrix[row - 1, col] != 0:
        vertical_clear = False  # 上方有试管
    if row < rows - 1 and matrix[row + 1, col] != 0:
        vertical_clear = False  # 下方有试管

    # 判断左右是否都没有试管
    horizontal_clear = True
    if col > 0 and matrix[row, col - 1] != 0:
        horizontal_clear = False  # 左侧有试管
    if col < cols - 1 and matrix[row, col + 1] != 0:
        horizontal_clear = False  # 右侧有试管

    # 只要上下或者左右没有试管,就可以抓取
    return vertical_clear or horizontal_clear


def find_robot_grabbable_positions(matrix):
    """找出所有满足机器人抓取条件的位置
    Args:
        matrix (np.ndarray): 试管架状态矩阵
    Returns:
        list: 包含所有可抓取位置的列表,每个位置表示为一个元组(row, col)
    """
    grabbable_positions = []
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 0 and is_robot_grabbable(matrix, r, c):
                grabbable_positions.append((r, c))
    return grabbable_positions


def simulate_grasp(matrix, position, grasper="Robot"):
    """模拟抓取动作,更新试管架状态
    Args:
        matrix (np.ndarray): 试管架状态矩阵
        position (tuple): 抓取位置(row, col)
        grasper (str): 抓取者,可以是"Robot"或"Human"
    Returns:
        np.ndarray: 更新后的试管架状态矩阵
    """
    row, col = position
    matrix[row, col] = 0  # 移除试管
    return matrix


def evaluate_human_grasp(matrix, row, col):
    """评估人工抓取指定位置对后续机器人抓取的影响(后续机器人可自主抓取的数量)
    Args:
        matrix (np.ndarray): 试管架状态矩阵
        row (int): 人工抓取位置的行索引
        col (int): 人工抓取位置的列索引
    Returns:
        int: 人工抓取后,机器人可自主抓取的试管数量
    """
    # 模拟人工抓取
    temp_matrix = matrix.copy()
    temp_matrix = simulate_grasp(temp_matrix, (row, col), "Human")
    # 统计机器人可抓取的位置数量
    robot_positions = find_robot_grabbable_positions(temp_matrix)
    return len(robot_positions)


def find_best_human_grasp_position(matrix):
    """找出最合适的人工抓取位置(最大化机器人后续可自主抓取的数量)
    Args:
        matrix (np.ndarray): 试管架状态矩阵
    Returns:
        tuple: 最合适的人工抓取位置(row, col),如果没有合适的位置,返回 None
    """
    best_position = None
    max_robot_positions = -1  # 初始值为-1,确保找到可抓取数量最多的位置
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 0:  # 只有有试管的位置才能抓取
                robot_positions = evaluate_human_grasp(matrix, r, c)  # 评估该位置的人工抓取效果
                if robot_positions > max_robot_positions:  # 如果该位置能带来更多的机器人可抓取数量
                    max_robot_positions = robot_positions
                    best_position = (r, c)
    return best_position
