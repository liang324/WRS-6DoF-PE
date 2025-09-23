# 包含表示动画的对象

from . import source
from .common import DaeObject
from .common import DaeError


class Animation(DaeObject):
    """
    用于保存来自 <animation> 标签的动画数据的类

    :param id: 动画的唯一标识符
    :param name: 动画的名称
    :param sourceById: 一个字典,包含所有源对象,以源的 ID 为键
    :param children: 子动画对象的列表
    :param xmlnode: 对应的 XML 节点,如果没有提供则为 None
    :return: 返回一个 `Animation` 对象,包含加载的动画数据
    """
    def __init__(self, id, name, sourceById, children, xmlnode=None):
        self.id = id
        self.name = name
        self.children = children
        self.sourceById = sourceById
        self.xmlnode = xmlnode
        if self.xmlnode is None:
            self.xmlnode = None

    @staticmethod
    def load(collada, localscope, node):
        """
        从给定的 XML 节点加载动画数据

        :param collada: COLLADA 文档对象
        :param localscope: 本地作用域,包含源对象的字典
        :param node: 包含动画数据的 XML 节点
        :return: 返回一个 `Animation` 对象,包含加载的动画数据
        """
        id = node.get('id') or ''
        name = node.get('name') or ''
        sourcebyid = localscope
        sources = []
        sourcenodes = node.findall(collada.tag('source'))
        for sourcenode in sourcenodes:
            ch = source.Source.load(collada, {}, sourcenode)
            sources.append(ch)
            sourcebyid[ch.id] = ch

        child_nodes = node.findall(collada.tag('animation'))
        children = []
        for child in child_nodes:
            try:
                child = Animation.load(collada, sourcebyid, child)
                children.append(child)
            except DaeError as ex:
                collada.handleError(ex)

        anim = Animation(id, name, sourcebyid, children, node)
        return anim

    def __str__(self):
        return '<Animation id=%s, children=%d>' % (self.id, len(self.children))

    def __repr__(self):
        return str(self)
