import time
import random
import networkx as nx
from motion.probabilistic import rrt


class RRTConnect(rrt.RRT):

    def __init__(self, robot_s):
        """
        初始化 RRTConnect类的实例
        :param robot_s: 机器人状态信息
        """
        super().__init__(robot_s)
        self.roadmap_start = nx.Graph()  # 起始路图
        self.roadmap_goal = nx.Graph()  # 目标路图

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
        :param roadmap: 当前路图
        :param conf: 当前配置
        :param ext_dist: 扩展距离
        :param goal_conf: 目标配置
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param animation: 是否显示动画
        :return: 如果成功连接到目标,返回'connection'；否则返回最近的节点ID
        author: weiwei
        date: 20201228
        """
        # 找到路图中与给定配置最近的节点ID
        nearest_nid = self._get_nearest_nid(roadmap, conf)

        # 从最近的节点开始,向目标配置方向扩展,生成新的配置列表
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist, exact_end=False)[1:]

        for new_conf in new_conf_list:
            # 检查新配置是否与障碍物或其他机器人发生碰撞
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return -1
            else:
                # 为新配置生成一个随机节点ID,并将其添加到路图中
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid

                # 如果启用动画,绘制工作空间
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf, '^c')
                # 检查是否到达目标
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # 添加连接节点
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        return nearest_nid

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        """
        对路径进行平滑处理,
        通过尝试路径的捷径来减少路径中的节点数量
        :param component_name: 组件名称
        :param path: 原始路径
        :param obstacle_list: 障碍物列表
        :param otherrobot_list: 其他机器人列表
        :param granularity: 平滑处理的粒度
        :param iterations: 平滑处理的迭代次数
        :param animation: 是否显示动画
        :return: 平滑处理后的路径
        """
        smoothed_path = path
        for _ in range(iterations):
            # 如果路径节点数小于等于2,直接返回路径
            if len(smoothed_path) <= 2:
                return smoothed_path
            # 随机选择路径中的两个节点
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            # 如果两个节点相邻,跳过本次迭代
            if abs(i - j) <= 1:
                continue
            # 确保i小于j
            if j < i:
                i, j = j, i
            # 尝试在两个节点之间创建捷径
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)
            # 如果捷径的长度小于等于原路径段,并且捷径上的所有配置都没有碰撞,则替换原路径段
            if (len(shortcut) <= (j - i) + 1) and all(not self._is_collided(component_name=component_name,
                                                                            conf=conf,
                                                                            obstacle_list=obstacle_list,
                                                                            otherrobot_list=otherrobot_list)
                                                      for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            # 如果启用动画,绘制工作空间
            if animation:
                self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
        return smoothed_path

    @rrt._decorator_keep_jnt_values
    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             max_iter=300,
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
            :param max_iter: 最大迭代次数
            :param max_time: 最大规划时间
            :param smoothing_iterations: 平滑处理的迭代次数
            :param animation: 是否显示动画
            :return: 平滑处理后的路径,如果未找到路径则返回 None

        """
        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf

        # 检查起始和目标配置是否与障碍物发生碰撞
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("起始配置与障碍物发生碰撞！")
            return None
        
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("目标配置与障碍物发生碰撞！")
            return None

        # 检查起始配置是否已经到达目标配置
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [start_conf, goal_conf]

        # 初始化起始和目标路图
        self.roadmap_start.add_node('start', conf=start_conf)
        self.roadmap_goal.add_node('goal', conf=goal_conf)

        tic = time.time()
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal
        tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
        tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']

        # 检查是否超过最大规划时间
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("规划时间过长！未能找到路径！")
                    return None

            # 使用随机目标扩展一棵树
            rand_conf = self._sample_conf(component_name=component_name,
                                          rand_rate=100,
                                          default_conf=None)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=tree_a,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=tree_a_goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid != -1:  # 如果没有陷入困境
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
                last_nid = self._extend_roadmap(component_name=component_name,
                                                roadmap=tree_b,
                                                conf=tree_a.nodes[last_nid]['conf'],
                                                ext_dist=ext_dist,
                                                goal_conf=tree_b_goal_conf,
                                                obstacle_list=obstacle_list,
                                                otherrobot_list=otherrobot_list,
                                                animation=animation)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
            # 如果树a的节点数大于树b,交换两棵树
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
        else:
            print("达到最大迭代次数！未能找到路径.")
            return None
        path = self._path_from_roadmap()
        # 从路图中提取路径并进行平滑处理
        smoothed_path = self._smooth_path(component_name=component_name,
                                          path=path,
                                          obstacle_list=obstacle_list,
                                          otherrobot_list=otherrobot_list,
                                          granularity=ext_dist,
                                          iterations=smoothing_iterations,
                                          animation=animation)
        return smoothed_path


# class RRTConnect_v2(rrt.RRT_v2):
#
#     def __init__(self, robot_s):
#         super().__init__(robot_s)
#         self.roadmap_start = nx.Graph()
#         self.roadmap_goal = nx.Graph()
#
#     def _extend_roadmap(self,
#                         roadmap,
#                         conf,
#                         ext_dist,
#                         goal_conf,
#                         obstacle_list=[],
#                         otherrobot_list=[],
#                         animation=False):
#         """
#         find the nearest point between the given roadmap and the conf and then extend towards the conf
#         :return:
#         author: weiwei
#         date: 20201228
#         """
#         nearest_nid = self._get_nearest_nid(roadmap, conf)
#         new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist, exact_end=False)[1:]
#         for new_conf in new_conf_list:
#             if self._is_collided(new_conf, obstacle_list, otherrobot_list):
#                 return -1
#             else:
#                 new_nid = random.randint(0, 1e16)
#                 roadmap.add_node(new_nid, conf=new_conf)
#                 roadmap.add_edge(nearest_nid, new_nid)
#                 nearest_nid = new_nid
#                 # all_sampled_confs.append([new_node.point, False])
#                 if animation:
#                     self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
#                                      obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf, '^c')
#                 # check goal
#                 if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
#                     roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
#                     roadmap.add_edge(new_nid, 'connection')
#                     return 'connection'
#         return nearest_nid
#
#     def _smooth_path(self,
#                      path,
#                      obstacle_list=[],
#                      otherrobot_list=[],
#                      granularity=2,
#                      iterations=50,
#                      animation=False):
#         smoothed_path = path
#         for _ in range(iterations):
#             if len(smoothed_path) <= 2:
#                 return smoothed_path
#             i = random.randint(0, len(smoothed_path) - 1)
#             j = random.randint(0, len(smoothed_path) - 1)
#             if abs(i - j) <= 1:
#                 continue
#             if j < i:
#                 i, j = j, i
#             shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)
#             if (len(shortcut) <= (j - i) + 1) and all(not self._is_collided(conf=conf,
#                                                                             obstacle_list=obstacle_list,
#                                                                             otherrobot_list=otherrobot_list)
#                                                       for conf in shortcut):
#                 smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
#             if animation:
#                 self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
#                                  obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
#         return smoothed_path
#
#     @rrt._decorator_keep_jnt_values2
#     def plan(self,
#              start_conf,
#              goal_conf,
#              obstacle_list=[],
#              otherrobot_list=[],
#              ext_dist=2,
#              max_iter=300,
#              max_time=15.0,
#              smoothing_iterations=50,
#              animation=False):
#         self.roadmap.clear()
#         self.roadmap_start.clear()
#         self.roadmap_goal.clear()
#         self.start_conf = start_conf
#         self.goal_conf = goal_conf
#         # check start and goal
#         if self._is_collided(start_conf, obstacle_list, otherrobot_list):
#             print("The start robot_s configuration is in collision!")
#             return None
#         if self._is_collided(goal_conf, obstacle_list, otherrobot_list):
#             print("The goal robot_s configuration is in collision!")
#             return None
#         if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
#             return [start_conf, goal_conf]
#         self.roadmap_start.add_node('start', conf=start_conf)
#         self.roadmap_goal.add_node('goal', conf=goal_conf)
#         tic = time.time()
#         tree_a = self.roadmap_start
#         tree_b = self.roadmap_goal
#         tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
#         tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']
#         for _ in range(max_iter):
#             toc = time.time()
#             if max_time > 0.0:
#                 if toc - tic > max_time:
#                     print("Too much motion time! Failed to find a path.")
#                     return None
#             # one tree grown using random target
#             rand_conf = self._sample_conf(rand_rate=100,
#                                           default_conf=None)
#             last_nid = self._extend_roadmap(roadmap=tree_a,
#                                             conf=rand_conf,
#                                             ext_dist=ext_dist,
#                                             goal_conf=tree_a_goal_conf,
#                                             obstacle_list=obstacle_list,
#                                             otherrobot_list=otherrobot_list,
#                                             animation=animation)
#             if last_nid != -1: # not trapped:
#                 goal_nid = last_nid
#                 tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
#                 last_nid = self._extend_roadmap(roadmap=tree_b,
#                                                 conf=tree_a.nodes[last_nid]['conf'],
#                                                 ext_dist=ext_dist,
#                                                 goal_conf=tree_b_goal_conf,
#                                                 obstacle_list=obstacle_list,
#                                                 otherrobot_list=otherrobot_list,
#                                                 animation=animation)
#                 if last_nid == 'connection':
#                     self.roadmap = nx.compose(tree_a, tree_b)
#                     self.roadmap.add_edge(last_nid, goal_nid)
#                     break
#                 elif last_nid != -1:
#                     goal_nid = last_nid
#                     tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
#             if tree_a.number_of_nodes() > tree_b.number_of_nodes():
#                 tree_a, tree_b = tree_b, tree_a
#                 tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
#         else:
#             print("Reach to maximum iteration! Failed to find a path.")
#             return None
#         path = self._path_from_roadmap()
#         smoothed_path = self._smooth_path(path=path,
#                                           obstacle_list=obstacle_list,
#                                           otherrobot_list=otherrobot_list,
#                                           granularity=ext_dist,
#                                           iterations=smoothing_iterations,
#                                           animation=animation)
#         return smoothed_path

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import robot_sim.robots.xybot.xybot as xyb

    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3),
        ((10, 0), 3),
        ((10, -2), 3),
        ((10, -4), 3),
        ((15, 5), 3),
        ((15, 7), 3),
        ((15, 9), 3),
        ((15, 11), 3),
        ((0, 12), 3),
        ((-1, 10), 3),
        ((-2, 8), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = xyb.XYBot()
    rrtc = RRTConnect(robot)
    path = rrtc.plan(start_conf=np.array([-2, 0]),
                     goal_conf=np.array([5, 10]),
                     obstacle_list=obstacle_list,
                     ext_dist=1,
                     max_time=300,
                     animation=True)
    # import time
    # total_t = 0
    # for i in range(100):
    #     tic = time.time()
    #     path = rrtc.plan(seed_jnt_values=np.array([0, 0]), end_conf=np.array([5, 10]), obs_list=obs_list,
    #                      ext_dist=1, rand_rate=70, max_time=300, hnd_name=None, animation=False)
    #     toc = time.time()
    #     total_t = total_t + toc - tic
    # print(total_t)
    # Draw final path
    print(path)
    rrtc.draw_wspace([rrtc.roadmap_start, rrtc.roadmap_goal],
                     rrtc.start_conf, rrtc.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=7, linestyle='-', color='c')
    # plt.savefig(str(rrtc.img_counter)+'.jpg')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
