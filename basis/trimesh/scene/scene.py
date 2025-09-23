import numpy as np
from ..points import transform_points
from ..grouping import group_rows
from ..util import is_sequence, is_instance_named
from ..transformations import rotation_matrix
from .transforms import TransformForest
from collections import deque


class Scene:
    """
    一个简单的场景图,可以直接通过 pyglet/openGL 渲染或通过其他端点(如光线追踪器)渲染
    网格和光源通过名称添加,然后可以通过更新变换树中的变换来移动.
    """
    def __init__(self, node=None, base_frame='world'):
        # 实例名称 : 网格名称
        self.nodes = {}
        # 网格名称 : Trimesh 对象
        self.meshes = {}
        self.flags = {}
        self.transforms = TransformForest(base_frame=base_frame)
        self.add_mesh(node)
        self.set_camera()

    def add_mesh(self, mesh):
        """
        将网格添加到场景中

        如果网格在其元数据中定义了多个变换则将在每个变换处创建网格的新实例
        """
        if is_sequence(mesh):
            for i in mesh:
                self.add_mesh(i)
            return
        if 'name' in mesh.metadata:
            name_mesh = mesh.metadata['name']
        else:
            name_mesh = 'mesh_' + str(len(self.meshes))
        self.meshes[name_mesh] = mesh
        if 'transforms' in mesh.metadata:
            transforms = np.array(mesh.metadata['transforms'])
        else:
            transforms = np.eye(4).reshape((-1, 4, 4))
        for i, transform in enumerate(transforms):
            name_node = name_mesh + '_' + str(i)
            self.nodes[name_node] = name_mesh
            self.flags[name_node] = {'visible': True}
            self.transforms.update(frame_to=name_node, matrix=transform)

    @property
    def bounds(self):
        """
        计算场景的整体边界

        :return: bounds 2x3 浮点数,表示最小和最大角落
        """
        corners = deque()
        for instance, mesh_name in self.nodes.items():
            transform = self.transforms.get(instance)
            corners.append(transform_points(self.meshes[mesh_name].bounds, transform))
        corners = np.vstack(corners)
        bounds = np.array([corners.min(axis=0), corners.max(axis=0)])
        return bounds

    @property
    def extents(self):
        """
        计算场景的范围

        :return: 场景的范围
        """
        return np.diff(self.bounds, axis=0).reshape(-1)

    @property
    def scale(self):
        """
        计算场景的缩放比例

        :return: 场景的最大范围
        """
        return self.extents.max()

    @property
    def centroid(self):
        """
        计算场景边界框的中心

        :return: centroid: (3) 浮点数,表示边界框的中心点
        """
        centroid = np.mean(self.bounds, axis=0)
        return centroid

    def duplicate_nodes(self):
        """
        返回节点键的序列,其中组中的所有键都将属于相同的网格

        :return: duplicates: 重复节点的数组
        """
        mesh_ids = {k: m.identifier for k, m in self.meshes.items()}
        node_keys = np.array(list(self.nodes.keys()))
        node_ids = [mesh_ids[v] for v in self.nodes.values()]
        node_groups = group_rows(node_ids, digits=1)
        duplicates = np.array([node_keys[g] for g in node_groups])
        return duplicates

    def set_camera(self, angles=None, distance=None, center=None):
        """
        设置相机的位置和方向

        :param angles: (3) 旋转角度,默认为
        :param distance: 相机到场景中心的距离,默认为边界框的最大差值
        :param center: 相机的中心点,默认为场景的质心
        """
        if center is None:
            center = self.centroid
        if distance is None:
            distance = np.diff(self.bounds, axis=0).max()
        if angles is None:
            angles = np.zeros(3)
        translation = np.eye(4)
        translation[0:3, 3] = center
        translation[2][3] += distance * 1.5
        transform = np.dot(rotation_matrix(angles[0], [1, 0, 0], point=center),
                           rotation_matrix(angles[1], [0, 1, 0], point=center))
        transform = np.dot(transform, translation)
        self.transforms.update(frame_from='camera', frame_to=self.transforms.base_frame, matrix=transform)

    def dump(self):
        """
        将场景中的所有网格附加到网格列表中

        :return: result: 网格的数组
        """
        result = deque()
        for node_id, mesh_id in self.nodes.items():
            transform = self.transforms.get(node_id)
            current = self.meshes[mesh_id].copy()
            current.transform(transform)
            result.append(current)
        return np.array(result)

    def export(self, file_type='dict64'):
        """
        导出当前场景的快照

        :param file_type: 用于网格的编码类型,例如:dict, dict64, meshes
        :return: export: 包含以下键的字典:
                 meshes: 网格列表,按 file_type 编码
                 transforms: 变换的边列表,例如:((u, v, {'matrix' : np.eye(4)}))
        """
        export = {}
        export['transforms'] = self.transforms.export()
        export['nodes'] = self.nodes
        export['meshes'] = {name: mesh.export(file_type) for name, mesh in self.meshes.items()}
        return export

    def save_image(self, file_obj, resolution=(1024, 768), **kwargs):
        """
        保存场景的图像

        :param file_obj: 文件对象,用于保存图像
        :param resolution: 图像的分辨率,默认为 (1024, 768)
        """
        from .viewer import SceneViewer
        SceneViewer(self, save_image=file_obj, resolution=resolution, **kwargs)

    def explode(self, vector=[0.0, 0.0, 1.0], origin=None):
        """
        围绕一个点和向量展开场景

        :param vector: (3,) 或 float,展开的方向或比例
        :param origin: 展开的原点,默认为场景的质心
        """
        if origin is None:
            origin = self.centroid
        centroids = np.array([self.meshes[i].centroid for i in self.nodes.values()])
        if np.shape(vector) == (3,):
            vectors = np.tile(vector, (len(centroids), 1))
            projected = np.dot(vector, (centroids - origin).T)
            offsets = vectors * projected.reshape((-1, 1))
        elif isinstance(vector, float):
            offsets = (centroids - origin) * vector
        else:
            raise ValueError('Explode vector must by (3,) or float')
        for offset, node_key in zip(offsets, self.nodes.keys()):
            current = self.transforms[node_key]
            current[0:3, 3] += offset
            self.transforms[node_key] = current

    def show(self, block=True, **kwargs):
        """
        显示场景

        :param block: 是否阻塞,默认为 True
        """
        # 这将导入pyglet,并将引发ImportError
        # 如果pyglet不可用
        from .viewer import SceneViewer
        def viewer():
            SceneViewer(self, **kwargs)

        if block:
            viewer()
        else:
            from threading import Thread
            Thread(target=viewer, kwargs=kwargs).start()
