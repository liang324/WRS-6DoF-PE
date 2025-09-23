import time
import math
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


# 定义RRTDW类,表示基于RRT的路径规划
class RRTDW(object):

    def __init__(self, robot_s):
        # 初始化函数,接受机器人状态对象
        self.robot_s = robot_s.copy()  # 复制机器人状态
        self.roadmap = nx.Graph()  # 创建空的图来存储路径
        self.start_conf = None  # 起始点配置
        self.goal_conf = None  # 目标点配置

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        """
        检查机器人在给定配置下是否与障碍物或其他机器人发生碰撞

        :param component_name: 机器人组件名称
        :param conf: 配置(机器人位置和姿态)
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :return: 如果碰撞返回True,否则返回False
        """
        self.robot_s.fk(component_name=component_name, jnt_values=conf)  # 计算正向运动学
        return self.robot_s.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)  # 判断是否碰撞

    def _sample_conf(self, component_name, rand_rate, default_conf):
        """
        随机采样配置

        :param component_name: 机器人组件名称
        :param rand_rate: 随机采样的概率
        :param default_conf: 默认配置
        :return: 随机采样得到的配置或默认配置
        """
        if random.randint(0, 99) < rand_rate:  # 按照采样概率选择
            return self.robot_s.rand_conf(component_name=component_name)  # 随机生成配置
        else:
            return default_conf  # 返回默认配置

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        获取与新配置最接近的节点ID

        :param roadmap: 当前的路径图
        :param new_conf: 新配置
        :return: 最接近节点的ID
        """
        nodes_dict = dict(roadmap.nodes(data='conf'))  # 获取图中所有节点的配置
        nodes_key_list = list(nodes_dict.keys())  # 节点的ID列表
        nodes_value_list = list(nodes_dict.values())  # 节点配置列表
        conf_array = np.array(nodes_value_list)  # 将配置列表转为NumPy数组
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)  # 计算新配置与图中每个节点的欧几里得距离
        min_dist_nid = np.argmin(diff_conf_array)  # 找到最小距离对应的节点ID
        return nodes_key_list[min_dist_nid]  # 返回最接近节点的ID

    def _extend_conf(self, conf1, conf2, ext_dist):
        """
        将conf1扩展到conf2之间,扩展步长为ext_dist

        :param conf1: 起始配置
        :param conf2: 目标配置
        :param ext_dist: 扩展距离
        :return: 从conf1到conf2的扩展路径
        """
        angle_ext_dist = ext_dist  # 角度扩展步长
        len, vec = rm.unit_vector(conf2[:2] - conf1[:2], toggle_length=True)  # 计算直线方向向量
        if len > 0:
            translational_theta = rm.angle_between_2d_vectors(np.array([1, 0]), vec)  # 计算平移角度
            conf1_theta_to_translational_theta = translational_theta - conf1[2]  # 计算当前角度与目标角度的差值
        else:
            conf1_theta_to_translational_theta = (conf2[2] - conf1[2])  # 如果长度为0,直接计算角度差
            translational_theta = conf2[2]

        # 旋转步长
        nval = abs(math.ceil(conf1_theta_to_translational_theta / angle_ext_dist))
        linear_conf1 = np.array([conf1[0], conf1[1], translational_theta])  # 创建一个线性配置
        conf1_angular_arary = np.linspace(conf1, linear_conf1, nval)  # 生成从conf1到线性配置的旋转步进
        # 平移步长
        nval = math.ceil(len / ext_dist)  # 根据长度计算平移步长
        linear_conf2 = np.array([conf2[0], conf2[1], translational_theta])  # 生成目标配置的线性表示
        conf12_linear_arary = np.linspace(linear_conf1, linear_conf2, nval)  # 从conf1线性过渡到conf2
        # 旋转步长
        translational_theta_to_conf2_theta = conf2[2] - translational_theta  # 计算平移角度到目标角度的差值
        nval = abs(math.ceil(translational_theta_to_conf2_theta / angle_ext_dist))  # 计算旋转步长
        conf2_angular_arary = np.linspace(linear_conf2, conf2, nval)  # 生成从conf2到目标配置的旋转步进
        conf_array = np.vstack((conf1_angular_arary, conf12_linear_arary, conf2_angular_arary))  # 将所有步进拼接
        return list(conf_array)  # 返回扩展路径

    def _extend_roadmap(self,
                        component_name,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        otherrobot_list=[],
                        animation=False):
        """
        扩展路径图,寻找最接近给定配置的节点并朝目标扩展

        :param component_name: 机器人组件名称
        :param roadmap: 路径图
        :param conf: 目标配置
        :param ext_dist: 扩展步长
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param animation: 是否显示动画
        :return: 最接近节点的ID或者'connection'(如果目标达成)
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)  # 获取最接近的节点ID
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)[1:]  # 扩展配置
        for new_conf in new_conf_list:
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return nearest_nid  # 如果发生碰撞,返回最接近节点ID
            else:
                new_nid = random.randint(0, 1e16)  # 生成一个新的节点ID
                roadmap.add_node(new_nid, conf=new_conf)  # 添加新的节点
                roadmap.add_edge(nearest_nid, new_nid)  # 添加新的边连接
                nearest_nid = new_nid  # 更新最近节点ID
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                     new_conf)  # 绘制动画
                # 检查是否到达目标
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # 如果目标达成,添加目标节点
                    roadmap.add_edge(new_nid, 'connection')  # 连接到目标节点
                    return 'connection'
        else:
            return nearest_nid  # 返回最接近节点ID

    def _goal_test(self, conf, goal_conf, threshold):
        """
        检查是否达到目标

        :param conf: 当前配置
        :param goal_conf: 目标配置
        :param threshold: 到达目标的阈值距离
        :return: 如果达到目标返回True,否则返回False
        """
        dist = np.linalg.norm(conf - goal_conf)  # 计算当前位置到目标位置的欧几里得距离
        if dist <= threshold:
            return True  # 如果距离小于阈值,说明达到了目标
        else:
            return False  # 否则未到达目标

    def _path_from_roadmap(self):
        """
        从路径图中提取路径

        :return: 路径点列表
        """
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')  # 使用最短路径算法获取路径
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))  # 提取路径中的配置点

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        """
        平滑路径

        :param component_name: 机器人组件名称
        :param path: 初始路径
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param granularity: 平滑步长
        :param iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: 平滑后的路径
        """
        smoothed_path = path  # 初始化平滑路径
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path  # 如果路径长度小于等于2,返回当前路径
            i = random.randint(0, len(smoothed_path) - 1)  # 随机选择路径点i
            j = random.randint(0, len(smoothed_path) - 1)  # 随机选择路径点j
            if abs(i - j) <= 1:
                continue  # 如果i和j相邻,则跳过
            if j < i:
                i, j = j, i  # 保证i小于j
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)  # 尝试生成一段快捷路径
            if all(not self._is_collided(component_name=component_name,
                                         conf=conf,
                                         obstacle_list=obstacle_list,
                                         otherrobot_list=otherrobot_list)
                   for conf in shortcut):  # 如果路径无碰撞
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]  # 更新平滑路径
            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)  # 显示动画
        return smoothed_path  # 返回平滑后的路径

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             rand_rate=70,
             max_iter=1000,
             max_time=15.0,
             smoothing_iterations=50,
             animation=False):
        """
        使用RRT算法规划路径

        :param component_name: 机器人组件名称
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param ext_dist: 扩展步长
        :param rand_rate: 随机采样的概率
        :param max_iter: 最大迭代次数
        :param max_time: 最大运行时间
        :param smoothing_iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: 返回平滑后的路径
        """
        self.roadmap.clear()  # 清空路径图
        self.start_conf = start_conf  # 设置起始配置
        self.goal_conf = goal_conf  # 设置目标配置
        # 检查起始配置和目标配置是否发生碰撞
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("起始配置与障碍物碰撞！")
            return None
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("目标配置与障碍物碰撞！")
            return None
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]  # 如果起始配置已经在目标附近,直接返回路径

        self.roadmap.add_node('start', conf=start_conf)  # 添加起始节点
        tic = time.time()  # 记录开始时间
        for _ in range(max_iter):
            toc = time.time()  # 记录当前时间
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("超时,无法找到路径.")
                    return None
            # 随机采样配置
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)  # 扩展路径图
            if last_nid == 'connection':  # 如果目标达成
                mapping = {'connection': 'goal'}  # 将目标节点命名为'goal'
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)  # 更新图的节点名称
                path = self._path_from_roadmap()  # 获取路径
                smoothed_path = self._smooth_path(component_name=component_name,
                                                  path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=smoothing_iterations,
                                                  animation=animation)  # 对路径进行平滑处理
                return smoothed_path  # 返回平滑后的路径
        else:
            print("达到最大迭代次数！无法找到路径.")
            return None

    @staticmethod
    def draw_robot(plt, conf, facecolor='grey', edgecolor='grey'):
        """
        绘制机器人

        :param plt: Matplotlib绘图对象
        :param conf: 配置(位置和姿态)
        :param facecolor: 机器人主体颜色
        :param edgecolor: 机器人边缘颜色
        """
        ax = plt.gca()
        x = conf[0]
        y = conf[1]
        theta = conf[2]
        ax.add_patch(plt.Circle((x, y), .5, edgecolor=edgecolor, facecolor=facecolor))  # 绘制机器人主体
        ax.add_patch(plt.Rectangle((x, y), .7, .1, math.degrees(theta), color='y'))  # 绘制机器人前进方向
        ax.add_patch(
            plt.Rectangle((x, y), -.1, .1, math.degrees(theta), edgecolor=edgecolor, facecolor=facecolor))  # 绘制机器人其他部分
        ax.add_patch(plt.Rectangle((x, y), .7, -.1, math.degrees(theta), color='y'))
        ax.add_patch(plt.Rectangle((x, y), -.1, -.1, math.degrees(theta), edgecolor=edgecolor, facecolor=facecolor))

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):
        """
        绘制工作空间

        :param roadmap_list: 路径图列表
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param near_rand_conf_pair: 目标点和随机采样点的连接线
        :param new_conf: 新配置
        :param shortcut: 平滑路径
        :param smoothed_path: 平滑路径
        :param delay_time: 动画延迟时间
        """
        plt.clf()  # 清空当前图
        ax = plt.gca()  # 获取绘图区域
        ax.set_aspect('equal', 'box')  # 设置坐标轴比例
        plt.grid(True)  # 显示网格
        plt.xlim(-4.0, 17.0)  # 设置x轴范围
        plt.ylim(-4.0, 17.0)  # 设置y轴范围

        # 绘制起点和目标点
        RRTDW.draw_robot(plt, start_conf, facecolor='r', edgecolor='r')  # 绘制起始机器人
        RRTDW.draw_robot(plt, goal_conf, facecolor='g', edgecolor='g')  # 绘制目标机器人

        # 绘制障碍物
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))  # 使用黑色圆形表示障碍物

        # 设置颜色列表用于绘制路径
        colors = 'bgrcmykw'

        # 绘制路径图
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                # 绘制每个节点
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o' + colors[i])

                # 绘制节点之间的边
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-' + colors[i])

        # 如果近距离随机点对存在,则绘制它们之间的连接线并显示它们的机器人状态
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")  # 使用虚线绘制连接线
            RRTDW.draw_robot(plt, near_rand_conf_pair[0], facecolor='grey', edgecolor='g')  # 绘制第一个随机点
            RRTDW.draw_robot(plt, near_rand_conf_pair[1], facecolor='grey', edgecolor='c')  # 绘制第二个随机点

        # 如果有新的配置点,则绘制该点
        if new_conf is not None:
            RRTDW.draw_robot(plt, new_conf, facecolor='grey', edgecolor='c')  # 使用灰色表示新点

        # 如果有平滑路径,则绘制平滑路径
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')  # 使用较粗的线绘制平滑路径

        # 如果有快捷路径,则绘制快捷路径
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')  # 使用红色虚线绘制快捷路径

        # 如果没有图像计数器,初始化它
        if not hasattr(RRTDW, 'img_counter'):
            RRTDW.img_counter = 0
        else:
            RRTDW.img_counter += 1

        # 设置延迟时间,控制图形更新的速度
        if delay_time > 0:
            plt.pause(delay_time)  # 暂停一段时间,以便查看图形

        # plt.waitforbuttonpress()  # 等待按键操作,如果需要可以启用这一行


if __name__ == '__main__':
    import robot_sim.robots.xybot.xybot as xyb

    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3)
    ]

    robot = xyb.XYTBot()
    rrtdw = RRTDW(robot)
    path = rrtdw.plan(start_conf=np.array([0, 0, 0]), goal_conf=np.array([6, 9, 0]), obstacle_list=obstacle_list,
                      ext_dist=1, rand_rate=70, max_time=300, component_name='all', animation=True)
    # plt.show()
    # nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    # plt.show()
    # import time
    # total_t = 0
    # for i in range(1):
    #     tic = time.time()
    #     path, sampledpoints = rrt.motion(obstaclelist=obstaclelist, animation=True)
    #     toc = time.time()
    #     total_t = total_t + toc - tic
    # print(total_t)
    # Draw final path
    print(path)
    rrtdw.draw_wspace([rrtdw.roadmap], rrtdw.start_conf, rrtdw.goal_conf, obstacle_list, delay_time=0)
    for conf in path:
        RRTDW.draw_robot(plt, conf, edgecolor='r')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
