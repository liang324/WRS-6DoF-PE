import basis.data_adapter as da
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CollisionChecker(object):
    """
    一个快速的碰撞检测器,最多支持32对碰撞检测

    作者: weiwei
    日期: 20201214 Osaka
    """

    def __init__(self, name="auto"):
        """
        初始化碰撞检测器,创建必要的组件
        """
        self.ctrav = CollisionTraverser()  # 碰撞遍历器,用于执行碰撞检测
        self.chan = CollisionHandlerQueue()  # 碰撞处理队列,用于管理碰撞事件
        self.np = NodePath(name)  # 碰撞节点路径,存储和管理碰撞节点
        # 创建一个碰撞掩码列表,用于区分不同的碰撞对象
        self.bitmask_list = [BitMask32(2 ** n) for n in range(31)]
        # 用于与外部非激活对象的碰撞检测
        self._bitmask_ext = BitMask32(2 ** 31)  # 31 为使用外部非活动物体进行 CD 做好准备

        # 存储所有碰撞元素的列表,便于快速访问
        self.all_cdelements = []  # 用于快速访问 cd 元素的 cdlnks 或 cdobjs 列表(cdlnks/cdobjs)

    def add_cdlnks(self, jlcobj, lnk_idlist):
        """
        将给定链接的碰撞节点附加到self.np,并清除它们的碰撞掩码

        当一个机器人被另一个机器人视为障碍物时,所有的碰撞元素的IntoCollideMask会设置为BitMask32(2**31),
        这样其他机器人可以将其active_cdelements与所有的碰撞元素进行比较.
        :param jlcobj: 机器人对象
        :param lnk_idlist: 链接ID列表

        作者: weiwei
        日期: 20201216 Toyonaka
        """
        for id in lnk_idlist:
            # 如果该链接的cdprimit_childid为-1,说明是第一次添加
            if jlcobj.lnks[id]['cdprimit_childid'] == -1:
                # 将链接的碰撞模型复制到当前的碰撞节点路径中,并清除掩码
                cdnp = jlcobj.lnks[id]['collision_model'].copy_cdnp_to(self.np, clearmask=True)
                # 将碰撞节点添加到碰撞遍历器
                self.ctrav.addCollider(cdnp, self.chan)
                # 将当前链接加入到所有碰撞元素列表中
                self.all_cdelements.append(jlcobj.lnks[id])
                # 更新该链接的cdprimit_childid
                jlcobj.lnks[id]['cdprimit_childid'] = len(self.all_cdelements) - 1
            else:
                # 如果该链接已经添加过,抛出异常
                raise ValueError("The link is already added!")

    def set_active_cdlnks(self, activelist):
        """
        设置指定的 碰撞链接 用于与外部障碍物的碰撞检测
        :param activelist: 激活的链接列表,实际上是像 [jlchain.lnk0, jlchain.lnk1...] 这样的列表,
                           对应的to list将在线的cd函数中进行设置
        作者: weiwei
        日期: 20201216 Toyonaka
        """
        for cdlnk in activelist:
            # 如果该链接的cdprimit_childid为-1,说明它尚未添加到碰撞器中
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the add_cdlnks function first!")
            # 获取链接对应的碰撞节点
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            # 设置该节点的碰撞掩码
            cdnp.node().setFromCollideMask(self._bitmask_ext)

    def set_cdpair(self, fromlist, intolist):
        """
        设置碰撞对,这些碰撞对将用于自碰撞检测

        :param fromlist: 碰撞源列表,格式为 [[bool, cdprimit_cache], ...],表示哪些物体发起碰撞
        :param intolist: 碰撞目标列表,格式为 [[bool, cdprimit_cache], ...],表示哪些物体接收碰撞
        :return: None
        """
        if len(self.bitmask_list) == 0:
            raise ValueError("Too many collision pairs! Maximum: 29")
        # 从bitmask列表中分配一个碰撞掩码
        allocated_bitmask = self.bitmask_list.pop()
        # 遍历 fromlist 中的所有碰撞节点,更新其 FromCollideMask(哪些物体会发起碰撞)
        for cdlnk in fromlist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_from_cdmask = cdnp.node().getFromCollideMask()
            new_from_cdmask = current_from_cdmask | allocated_bitmask
            cdnp.node().setFromCollideMask(new_from_cdmask)

        # 遍历 intolist 中的所有碰撞节点,更新其 IntoCollideMask(哪些物体会接收碰撞)
        for cdlnk in intolist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("首先需要使用addjlcobj函数将链接添加到collider!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | allocated_bitmask
            cdnp.node().setIntoCollideMask(new_into_cdmask)

    def add_cdobj(self, objcm, rel_pos, rel_rotmat, into_list):
        """
        将碰撞对象添加到碰撞检测系统,并返回该对象的相关信息

        :return: cdobj_info,一个字典,包含与关节链接类似的信息,此外还包含 'into_list' 关键字,便于轻松关闭碰撞掩码
        """
        cdobj_info = {}
        cdobj_info['collision_model'] = objcm  # 用于反向查找碰撞模型
        cdobj_info['gl_pos'] = objcm.get_pos()  # 获取碰撞对象的位置
        cdobj_info['gl_rotmat'] = objcm.get_rotmat()  # 获取碰撞对象的旋转矩阵
        cdobj_info['rel_pos'] = rel_pos  # 获取物体相对位置
        cdobj_info['rel_rotmat'] = rel_rotmat  # 获取物体相对旋转矩阵
        cdobj_info['into_list'] = into_list  # 获取与该物体发生碰撞的目标物体列表

        # 将碰撞模型复制到当前节点路径,并清除掩码
        cdnp = objcm.copy_cdnp_to(self.np, clearmask=True)
        cdnp.node().setFromCollideMask(self._bitmask_ext)  # 激活碰撞掩码
        # 将碰撞对象添加到碰撞遍历器中
        self.ctrav.addCollider(cdnp, self.chan)
        # 保存碰撞对象的信息
        self.all_cdelements.append(cdobj_info)
        cdobj_info['cdprimit_childid'] = len(self.all_cdelements) - 1
        # 设置碰撞对
        self.set_cdpair([cdobj_info], into_list)
        return cdobj_info

    def delete_cdobj(self, cdobj_info):
        """
        删除指定的碰撞对象,并从碰撞检测系统中移除它

        :param cdobj_info: 一个类似关节链接的物体,由 self.add_cdobj 生成
        :return: None
        """
        # 从碰撞元素列表中删除该碰撞对象
        self.all_cdelements.remove(cdobj_info)
        # 获取要删除的碰撞节点
        cdnp_to_delete = self.np.getChild(cdobj_info['cdprimit_childid'])
        # 从碰撞遍历器中移除该碰撞节点
        self.ctrav.removeCollider(cdnp_to_delete)
        # 获取当前的碰撞掩码
        this_cdmask = cdnp_to_delete.node().getFromCollideMask()
        this_cdmask_exclude_ext = this_cdmask & ~self._bitmask_ext
        # 更新与其他碰撞元素的碰撞关系
        for cdlnk in cdobj_info['into_list']:
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask & ~this_cdmask_exclude_ext
            cdnp.node().setIntoCollideMask(new_into_cdmask)
        # 删除节点
        cdnp_to_delete.detachNode()
        # 将碰撞掩码返回到可用列表
        self.bitmask_list.append(this_cdmask_exclude_ext)

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contact_points=False):
        """
        检测是否发生碰撞

        :param obstacle_list: 静态几何模型的列表
        :param otherrobot_list: 其他机器人列表
        :param toggle_contact_points: 是否返回接触点
        :return: 碰撞结果(True 或 False),以及(如果需要)接触点列表
        """
        # 遍历所有的碰撞元素
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']  # 获取碰撞元素的全局位置
            rotmat = cdelement['gl_rotmat']  # 获取碰撞元素的全局旋转矩阵
            cdnp = self.np.getChild(cdelement['cdprimit_childid'])  # 获取碰撞节点

            # 设置节点的位置和旋转
            cdnp.setPosQuat(da.npv3_to_pdv3(pos), da.npmat3_to_pdquat(rotmat))

            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.ctrav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())

        # 附加障碍物
        obstacle_parent_list = []
        for obstacle in obstacle_list:
            obstacle_parent_list.append(obstacle.objpdnp.getParent())  # 保存原父节点
            obstacle.objpdnp.reparentTo(self.np)  # 将障碍物挂到当前节点

        # 附加其他机器人
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                # 更新与碰撞相关的掩码
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask | self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.reparentTo(self.np)  # 将其他机器人挂到当前节点

        # 执行碰撞检测
        self.ctrav.traverse(self.np)

        # 清除障碍物
        for i, obstacle in enumerate(obstacle_list):
            obstacle.objpdnp.reparentTo(obstacle_parent_list[i])  # 恢复障碍物的父节点

        # 清除其他机器人
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                # 恢复碰撞掩码
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask & ~self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.detachNode()  # 从场景树中移除机器人

        # 根据碰撞通道的数量判断是否发生了碰撞
        if self.chan.getNumEntries() > 0:
            collision_result = True
        else:
            collision_result = False

        # 如果需要返回接触点
        if toggle_contact_points:
            contact_points = [da.pdv3_to_npv3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                              self.chan.getEntries()]
            return collision_result, contact_points  # 返回碰撞结果和接触点
        else:
            return collision_result  # 仅返回碰撞结果

    def show_cdprimit(self):
        """
        复制当前节点路径到 base.render,并显示碰撞体的状态

        :return: None
        """
        # 复制节点到 base.render 以便显示碰撞体
        snp_cpy = self.np.copyTo(base.render)
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']  # 获取碰撞元素的位置
            rotmat = cdelement['gl_rotmat']  # 获取碰撞元素的旋转矩阵
            cdnp = snp_cpy.getChild(cdelement['cdprimit_childid'])  # 获取碰撞节点
            cdnp.setPosQuat(da.npv3_to_pdv3(pos), da.npmat3_to_pdquat(rotmat))  # 设置节点位置和旋转
            cdnp.show()  # 显示碰撞节点

    def disable(self):
        """
        清除所有碰撞对和节点路径

        :return: None
        """
        # 清除所有碰撞元素的子ID
        for cdelement in self.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.all_cdelements = []  # 清空所有碰撞元素
        # 从场景树中移除所有子节点
        for child in self.np.getChildren():
            child.removeNode()
        # 恢复碰撞掩码列表
        self.bitmask_list = list(range(31))  # 恢复最大 31 个掩码


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
