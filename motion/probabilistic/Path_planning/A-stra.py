# -*- coding: utf-8 -*-
"""
Author: Yixuan Su
Date: 2025/05/14 15:51
File: A-stra.py
Description:
"""

import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # 从起点到当前节点的代价
        self.h = 0  # 从当前节点到终点的估计代价(启发式)
        self.f = 0  # 总代价

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    # 使用曼哈顿距离作为启发式函数
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, end):
    open_list = []
    closed_list = set()

    start_node = Node(start)
    end_node = Node(end)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            if neighbor_pos[0] < 0 or neighbor_pos[0] >= len(grid) or neighbor_pos[1] < 0 or neighbor_pos[1] >= len(
                    grid[0]):
                continue

            if grid[neighbor_pos[0]][neighbor_pos[1]] != 0 or neighbor_pos in closed_list:
                continue

            neighbor_node = Node(neighbor_pos, current_node)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor_pos, end_node.position)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if any(open_node.position == neighbor_node.position and open_node.g <= neighbor_node.g for open_node in
                   open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return None


# 示例网格 ,0 表示可通行 ,1 表示障碍物
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
end = (4, 4)

path = astar(grid, start, end)
print("Path:", path)
