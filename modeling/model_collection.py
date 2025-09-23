import basis.robot_math as rm
import modeling._ode_cdhelper as mcd


class ModelCollection(object):
    """
    用于管理和可视化一组碰撞模型和几何模型

    author: weiwei
    date: 201900825, 20201212
    """

    def __init__(self, name='modelcollection'):
        """
        初始化 ModelCollection 实例

        :param name: 模型集合的名称,默认为 'modelcollection'
        """
        self._name = name
        self._gm_list = []  # 存储几何模型的列表
        self._cm_list = []  # 存储碰撞模型的列表

    @property
    def name(self):
        """
        获取模型集合的名称

        :return: 模型集合的名称
        """
        return self._name

    @property
    def cm_list(self):
        """
        获取碰撞模型列表

        :return: 碰撞模型列表
        """
        return self._cm_list

    @property
    def gm_list(self):
        """
        获取几何模型列表

        :return: 几何模型列表
        """
        return self._gm_list

    @property
    def cdmesh(self):
        """
        生成并返回碰撞网格

        根据不同的类型选择不同的网格类型(AABB、OBB、凸包或三角形).
        使用齐次变换矩阵对顶点和法线进行变换.

        :return: 生成的碰撞网格
        """
        vertices = []
        vertex_normals = []
        faces = []
        for objcm in self._cm_list:
            if objcm.cdmesh_type == 'aabb':
                objtrm = objcm.objtrm.bounding_box
            elif objcm.cdmesh_type == 'obb':
                objtrm = objcm.objtrm.bounding_box_oriented
            elif objcm.cdmesh_type == 'convexhull':
                objtrm = objcm.objtrm.convex_hull
            elif objcm.cdmesh_type == 'triangles':
                objtrm = objcm.objtrm

            homomat = objcm.get_homomat()
            vertices += rm.homomat_transform_points(homomat, objtrm.vertices)
            vertex_normals += rm.homomat_transform_points(homomat, objtrm.vertex_normals)
            faces += (objtrm.faces + len(faces))
        return mcd.gen_cdmesh_vvnf(vertices, vertex_normals, faces)

    @property
    def cdmesh_list(self):
        """
        获取所有碰撞模型的碰撞网格列表

        :return: 碰撞网格列表
        """
        return [objcm.cdmesh for objcm in self._cm_list]

    def add_cm(self, objcm):
        """
        将一个碰撞模型添加到 _cm_list 中

        :param objcm: 要添加的碰撞模型
        """
        self._cm_list.append(objcm)

    def remove_cm(self, objcm):
        """
        从 _cm_list 中移除一个碰撞模型

        :param objcm: 要移除的碰撞模型
        """
        self._cm_list.remove(objcm)

    def add_gm(self, objcm):
        """
        将一个几何模型添加到 _gm_list 中

        :param objcm: 要添加的几何模型
        """
        self._gm_list.append(objcm)

    def remove_gm(self, objcm):
        """
        从 _gm_list 中移除一个几何模型

        :param objcm: 要移除的几何模型
        """
        self._gm_list.remove(objcm)

    def attach_to(self, obj):
        """
        将所有的碰撞模型和几何模型附加到指定的对象 obj 上

        :param obj: 要附加模型的目标对象
        """
        # TODO: 检查 obj 是否为 ShowBase 实例
        for cm in self._cm_list:
            cm.attach_to(obj)
        for gm in self._gm_list:
            gm.attach_to(obj)

    def detach(self):
        """
        从所有的碰撞模型和几何模型中移除附加的对象
        """
        for cm in self._cm_list:
            cm.detach()
        for gm in self._gm_list:
            gm.detach()

    def remove(self):
        """
        从场景中移除所有碰撞模型和几何模型
        """
        for cm in self._cm_list:
            cm.remove()
        for gm in self._gm_list:
            gm.remove()

    def show_cdprimit(self):
        """
        显示所有碰撞模型的碰撞原语
        """
        for cm in self._cm_list:
            cm.show_cdprimit()

    def unshow_cdprimit(self):
        """
        隐藏所有碰撞模型的碰撞原语
        """
        for cm in self._cm_list:
            cm.unshow_cdprimit()

    def show_cdmesh(self):
        """
        显示所有碰撞模型的碰撞网格
        """
        for objcm in self._cm_list:
            objcm.show_cdmesh()

    def unshow_cdmesh(self):
        """
        隐藏所有碰撞模型的碰撞网格
        """
        for objcm in self._cm_list:
            objcm.unshow_cdmesh()
