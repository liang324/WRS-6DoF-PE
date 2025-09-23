import time
import math
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


def _decorator_keep_jnt_values(foo):
    """
    装饰器函数,用于保存和恢复机器人关节的值
    :param foo: 被装饰的函数
    :return:
    author: weiwei
    date: 20220404
    """

    def wrapper(self, component_name, *args, **kwargs):
        # 保存当前组件的关节值
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # 执行被装饰的函数
        result = foo(self, component_name, *args, **kwargs)
        # 恢复保存的关节值
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return result

    return wrapper


def _decorator_keep_jnt_values2(foo):
    """
    装饰器函数,用于保存和恢复机器人所有关节的值
    :param foo: 被装饰的函数
    :return: 装饰后的函数
    date: 20220404
    """

    def wrapper(self, *args, **kwargs):
        # 保存当前所有关节的值
        jnt_values_bk = self.robot_s.get_jnt_values()
        # 执行被装饰的函数
        result = foo(self, *args, **kwargs)
        # 恢复保存的关节值
        self.robot_s.fk(jnt_values=jnt_values_bk)
        return result

    return wrapper


class RRT(object):
    def __init__(self, robot_s):
        """
        初始化RRT类的实例
        :param robot_s: 机器人状态信息
        """
        self.robot_s = robot_s
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        """
        首先检查给定配置的关节值是否在范围内,如果任何关节值超出范围,将立即返回False,否则,将计算正向运动学并进行碰撞检测
        :param component_name: 组件名称
        :param conf: 配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :return: 如果发生碰撞返回True,否则返回False
        author: weiwei
        date: 20220326
        """
        # self.robot_s.fk(component_name=component_name, jnt_values=conf)
        # return self.robot_s.is_collided(obs_list=obs_list, otherrobot_list=otherrobot_list)

        if self.robot_s.is_jnt_values_in_ranges(component_name=component_name, jnt_values=conf):
            self.robot_s.fk(component_name=component_name, jnt_values=conf)
            return self.robot_s.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        else:
            print("给定的关节角度超出了关节限制.")
            return True

    def _sample_conf(self, component_name, rand_rate, default_conf):
        """
        采样一个新的配置
        :param component_name: 组件名称
        :param rand_rate: 随机率
        :param default_conf: 默认配置
        :return: 新的配置
        """
        if random.randint(0, 99) < rand_rate:
            return self.robot_s.rand_conf(component_name=component_name)
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        获取与新配置最近的节点ID
        :param roadmap: 路图
        :param new_conf: 新的配置
        :return: 最近的节点ID
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data='conf'))
        nodes_key_list = list(nodes_dict.keys())
        nodes_value_list = list(nodes_dict.values())  # 注意,Python中不保证对应关系

        # 如果通信不好(稍慢),使用以下替代方案,20210523,weiwei
        # # nodes_value_list = list(nodes_dict.values())
        # nodes_value_list = itemgetter(*nodes_key_list)(nodes_dict)
        # if type(nodes_value_list) == np.ndarray:
        #     nodes_value_list = [nodes_value_list]

        conf_array = np.array(nodes_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)  # 找到 diff_conf_array 中最小值的索引
        return nodes_key_list[min_dist_nid]

    def _extend_conf(self, conf1, conf2, ext_dist, exact_end=True):
        """
        扩展配置之间的路径
        :param conf1: 起始配置
        :param conf2: 目标配置
        :param ext_dist: 扩展距离
        :param exact_end: 是否精确到达终点
        :return: 配置列表
        """
        len, vec = rm.unit_vector(conf2 - conf1, toggle_length=True)
        # 一步扩展:未采用,因为它比完整扩展慢,20210523,weiwei
        # return [conf1 + ext_dist * vec] 切换到以下代码以获得完整的扩展
        if not exact_end:
            nval = math.ceil(len / ext_dist)  # 返回大于或等于输入值的最小整数
            nval = 1 if nval == 0 else nval  # 至少包括自身
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(len / ext_dist)  # 返回小于或等于输入值的最大整数
            nval = 1 if nval == 0 else nval  # 至少包括自身
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, conf2))
        return list(conf_array)

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
        在给定的路图和配置之间找到最近的点,然后向配置方向扩展
        :param component_name: 组件名称
        :param roadmap: 路图
        :param conf: 当前配置
        :param ext_dist: 扩展距离
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param animation: 是否显示动画
        :return: 如果连接到目标返回'connection',否则返回最近节点ID
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)

        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)[1:]

        for new_conf in new_conf_list:
            # 检查新配置是否与障碍物或其他机器人发生碰撞
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return nearest_nid
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid

                # all_sampled_confs.append([new_node.point, False])

                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                     new_conf, '^c')

                # 检查是否到达目标
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        else:
            return nearest_nid

    def _goal_test(self, conf, goal_conf, threshold):
        """
        检查当前配置是否到达目标配置

        :param conf: 当前配置
        :param goal_conf: 目标配置
        :param threshold: 阈值距离
        :return: 如果到达目标返回True,否则返回False
        """
        dist = np.linalg.norm(conf - goal_conf)
        if dist <= threshold:
            # print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        """
        从路图中提取路径

        :return: 路径列表
        """
        # self.roadmap 是一个图结构,表示整个路径规划的搜索空间    nid_path 是一个列表,包含最短路径上节点的 ID
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        # itemgetter 函数从这个字典中提取 nid_path 中每个节点的配置
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        """
        对路径进行平滑处理

        :param component_name: 组件名称
        :param path: 原始路径
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param granularity: 平滑粒度
        :param iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: 平滑后的路径
        """
        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path

            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)

            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i

            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)

            # 20210523,似乎我们不需要检查线长
            # if (len(shortcut) <= (j - i)) and all(not self._is_collided(component_name=component_name,
            #                                                            conf=conf,
            #                                                            obs_list=obs_list,
            #                                                            otherrobot_list=otherrobot_list)
            #                                      for conf in shortcut):

            if all(not self._is_collided(component_name=component_name,
                                         conf=conf,
                                         obstacle_list=obstacle_list,
                                         otherrobot_list=otherrobot_list)
                   for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]

            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
        return smoothed_path

    @_decorator_keep_jnt_values
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
        规划从起始配置到目标配置的路径

        :param component_name: 组件名称
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param ext_dist: 扩展距离
        :param rand_rate: 随机采样率
        :param max_iter: 最大迭代次数
        :param max_time: 最大时间限制
        :param smoothing_iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: [path, all_sampled_confs]

        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf

        # 检查起始和目标配置是否与障碍物或其他机器人发生碰撞
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("起始配置与障碍物发生碰撞!")
            return None

        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("目标配置与障碍物发生碰撞!")
            return None

        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]

        self.roadmap.add_node('start', conf=start_conf)
        tic = time.time()
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("规划时间过长! 未能找到路径.")
                    return None

            # 随机采样
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)

            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                # 从更新后的路图中提取路径
                path = self._path_from_roadmap()

                # 对提取的路径进行平滑处理
                smoothed_path = self._smooth_path(component_name=component_name,
                                                  path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=smoothing_iterations,
                                                  animation=animation)
                return smoothed_path
        else:
            print("达到最大迭代次数! 未能找到路径.")
            return None

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    new_conf_mark='^r',
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):
        """
        绘制工作空间图

        :param roadmap_list: 路图列表
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param near_rand_conf_pair: 最近随机配置对
        :param new_conf: 新配置
        :param new_conf_mark: 新配置标记
        :param shortcut: 捷径
        :param smoothed_path: 平滑路径
        :param delay_time: 延迟时间
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        ax.add_patch(plt.Circle((start_conf[0], start_conf[1]), .5, color='r'))
        ax.add_patch(plt.Circle((goal_conf[0], goal_conf[1]), .5, color='g'))
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = 'bgrcmykw'
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-' + colors[i])
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
            ax.add_patch(plt.Circle((near_rand_conf_pair[1][0], near_rand_conf_pair[1][1]), .3, color='grey'))
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        # plt.plot(planner.seed_jnt_values[0], planner.seed_jnt_values[1], "xr")
        # plt.plot(planner.end_conf[0], planner.end_conf[1], "xm")
        if not hasattr(RRT, 'img_counter'):
            RRT.img_counter = 0
        else:
            RRT.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


class RRT_v2(object):
    """
    这个版本不涉及组件名称；
    它是为 robot_sim.robots.arm_interface 的实例设计的
    author: weiwei
    date: 20230807
    """

    def __init__(self, robot_s):
        """
        初始化 RRT_v2 类的实例

        :param robot_s: 机器人状态信息
        """
        self.robot_s = robot_s
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None

    def _is_collided(self,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        """
        检查给定配置的关节值是否在范围内,如果任何关节值超出范围,将立即返回 False,否则,将计算正向运动学并进行碰撞检测

        :param conf: 配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :return: 是否发生碰撞
        author: weiwei
        date: 20220326
        """
        if self.robot_s.is_jnt_values_in_ranges(jnt_values=conf):
            self.robot_s.fk(jnt_values=conf)
            return self.robot_s.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        else:
            print("给定的关节角度超出了关节限制.")
            return True

    def _sample_conf(self, rand_rate, default_conf):
        """
        根据随机率采样配置

        :param rand_rate: 随机率
        :param default_conf: 默认配置
        :return: 采样的配置
        """
        if random.randint(0, 99) < rand_rate:
            return self.robot_s.rand_conf()
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        获取路图中与新配置最近的节点ID

        :param roadmap: 路图
        :param new_conf: 新配置
        :return: 最近的节点ID
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data='conf'))
        nodes_key_list = list(nodes_dict.keys())
        nodes_value_list = list(nodes_dict.values())  # 注意,在Python中对应关系不一定保证
        # 如果对应关系不好,可以使用以下替代方法(稍慢),20210523,weiwei
        # # nodes_value_list = list(nodes_dict.values())
        # nodes_value_list = itemgetter(*nodes_key_list)(nodes_dict)
        # if type(nodes_value_list) == np.ndarray:
        #     nodes_value_list = [nodes_value_list]
        conf_array = np.array(nodes_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)
        return nodes_key_list[min_dist_nid]

    def _extend_conf(self, conf1, conf2, ext_dist, exact_end=True):
        """
        扩展配置,从 conf1 向 conf2 延伸

        :param conf1: 起始配置
        :param conf2: 目标配置
        :param ext_dist: 扩展距离
        :param exact_end: 是否精确到达终点
        :return: 1xn 的 numpy 数组列表
        """
        len, vec = rm.unit_vector(conf2 - conf1, toggle_length=True)
        # 单步扩展:未采用,因为它比完全扩展慢,20210523,weiwei
        # return [conf1 + ext_dist * vec]
        # 切换到以下代码进行完全扩展
        if not exact_end:
            nval = math.ceil(len / ext_dist)
            nval = 1 if nval == 0 else nval  # 至少包括自身
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(len / ext_dist)
            nval = 1 if nval == 0 else nval  # 至少包括自身
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, conf2))
        return list(conf_array)

    def _extend_roadmap(self,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        otherrobot_list=[],
                        animation=False):
        """
        在给定的路图和配置之间找到最近的点,然后向配置方向扩展

        :param roadmap: 路图
        :param conf: 配置
        :param ext_dist: 扩展距离
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param animation: 是否显示动画
        :return: 最近的节点ID或'connection'
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, obstacle_list, otherrobot_list):
                return nearest_nid
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                     new_conf, '^c')
                # 检查是否到达目标
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        else:
            return nearest_nid

    def _goal_test(self, conf, goal_conf, threshold):
        """
        检查当前配置是否到达目标配置

        :param conf: 当前配置
        :param goal_conf: 目标配置
        :param threshold: 阈值
        :return: 如果到达目标则返回True,否则返回False
        """
        dist = np.linalg.norm(conf - goal_conf)
        if dist <= threshold:
            # print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        """
        从路图中提取路径

        :return: 路径配置列表
        """
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))

    def _smooth_path(self,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        """
        对路径进行平滑处理

        :param path: 原始路径
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param granularity: 平滑的粒度
        :param iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: 平滑后的路径
        """
        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)

            # 20210523, it seems we do not need to check line length
            # if (len(shortcut) <= (j - i)) and all(not self._is_collided(component_name=component_name,
            #                                                            conf=conf,
            #                                                            obs_list=obs_list,
            #                                                            otherrobot_list=otherrobot_list)
            #                                      for conf in shortcut):
            if all(not self._is_collided(conf=conf,
                                         obstacle_list=obstacle_list,
                                         otherrobot_list=otherrobot_list)
                   for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
        return smoothed_path

    @_decorator_keep_jnt_values
    def plan(self,
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
        规划路径,从起始配置到目标配置

        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param ext_dist: 扩展距离
        :param rand_rate: 随机采样率
        :param max_iter: 最大迭代次数
        :param max_time: 最大时间
        :param smoothing_iterations: 平滑迭代次数
        :param animation: 是否显示动画
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # 检查 seed_jnt_values和end_conf
        if self._is_collided(start_conf, obstacle_list, otherrobot_list):
            print("起始机器人配置发生碰撞!")
            return None
        if self._is_collided(goal_conf, obstacle_list, otherrobot_list):
            print("目标机器人配置发生碰撞!")
            return None
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]
        self.roadmap.add_node('start', conf=start_conf)
        tic = time.time()
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("运动时间过长！未能找到路径.")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=smoothing_iterations,
                                                  animation=animation)
                return smoothed_path
        else:
            print("达到最大迭代次数！未能找到路径.")
            return None

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    new_conf_mark='^r',
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):
        """
        绘制工作空间图

        :param roadmap_list: 路图列表
        :param start_conf: 起始配置
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param near_rand_conf_pair: 最近随机配置对
        :param new_conf: 新配置
        :param new_conf_mark: 新配置标记
        :param shortcut: 捷径路径
        :param smoothed_path: 平滑路径
        :param delay_time: 延迟时间
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        # 绘制起始和目标配置
        ax.add_patch(plt.Circle((start_conf[0], start_conf[1]), .5, color='r'))
        ax.add_patch(plt.Circle((goal_conf[0], goal_conf[1]), .5, color='g'))
        # 绘制障碍物
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        # 绘制路图
        colors = 'bgrcmykw'
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-' + colors[i])
        # 绘制最近随机配置对
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
            ax.add_patch(plt.Circle((near_rand_conf_pair[1][0], near_rand_conf_pair[1][1]), .3, color='grey'))
        # 绘制新配置
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        # 绘制平滑路径
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        # 绘制捷径路径
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        # plt.plot(planner.seed_jnt_values[0], planner.seed_jnt_values[1], "xr")
        # plt.plot(planner.end_conf[0], planner.end_conf[1], "xm")
        # 保存图片计数器
        if not hasattr(RRT, 'img_counter'):
            RRT.img_counter = 0
        else:
            RRT.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        # 延迟显示
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    import robot_sim.robots.xybot.xybot as xyb

    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = xyb.XYBot()
    rrt = RRT(robot)
    component_name = 'all'
    path = rrt.plan(component_name=component_name, start_conf=np.array([0, 0]), goal_conf=np.array([6, 9]),
                    obstacle_list=obstacle_list,
                    ext_dist=1, rand_rate=70, max_time=300, animation=True)
    # plt.show()
    nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    plt.show()

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
    rrt.draw_wspace([rrt.roadmap], rrt.start_conf, rrt.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=4, color='c')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
