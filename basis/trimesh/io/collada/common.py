from .xmlutil import etree, ElementMaker, COLLADA_NS

E = ElementMaker(namespace=COLLADA_NS, nsmap={None: COLLADA_NS})


def tag(text, namespace=None):
    """
    使用 COLLADA 命名空间标记文本键,默认命名空间为:
    '{http://www.collada.org/2005/11/COLLADASchema}'

    :param string text:要标记的文本,例如 'geometry'
    :param string namespace:要使用的命名空间(不包括括号)如果传递 None,将使用默认命名空间
    """
    if namespace is None:
        namespace = COLLADA_NS
    return str(etree.QName(namespace, text))


def tagger(namespace=None):
    """
    一个闭包,或返回一个函数的函数,返回的函数使用指定的命名空间进行标记

    :param string namespace:用于标记元素的 XML 命名空间

    :return: tag() 函数
    """

    def tag(text):
        return str(etree.QName(namespace, text))

    return tag


class DaeObject(object):
    """
    这个类是所有 COLLADA 对象的抽象接口

    COLLADA 中我们识别和加载的每个 <tag> 都有一个从这个类派生的镜像类.
    所有实例至少有一个 :meth:`load` 方法,该方法从 XML 节点创建对象,
    以及一个名为 :attr:`xmlnode` 的属性,表示数据的 ElementTree 表示.
    即使它是动态创建的.如果对象不是只读的,它还将有一个 :meth:`save` 方法,
    该方法将对象的信息保存回 :attr:`xmlnode` 属性.
    """
    xmlnode = None
    """ElementTree representation of the log_filename."""

    @staticmethod
    def load(collada, localscope, node):
        """
        从 XML 节点加载并返回类实例
        检查节点内的数据,该数据必须与此类标签匹配,并从中创建实例

        :param collada.Collada collada: 该对象所在的 COLLADA 文件对象
        :param dict localscope: 如果有一个本地作用域,我们应该在其中查找本地 ID (sid),这是字典.否则为空字典 ({})
        :param node: 来自 Python 的 ElementTree API 的元素
        """
        raise Exception('Not implemented')

    def save(self):
        """
        将所有数据放入内部 XML 节点 (xmlnode) 以便可以序列化
        """


class DaeError(Exception):
    """
    一般 DAE 异常
    """

    def __init__(self, msg):
        super(DaeError, self).__init__()
        self.msg = msg

    def __str__(self):
        return type(self).__name__ + ': ' + self.msg

    def __repr__(self):
        return type(self).__name__ + '("' + self.msg + '")'


class DaeIncompleteError(DaeError):
    """
    当对象所需的数据不存在时引发
    """
    pass


class DaeBrokenRefError(DaeError):
    """
    当在作用域中找不到引用的对象时引发
    """
    pass


class DaeMalformedError(DaeError):
    """Raised when log_filename is found to be corrupted in some way."""
    pass


class DaeUnsupportedError(DaeError):
    """Raised when some unexpectedly unsupported feature is found."""
    pass


class DaeSaveValidationError(DaeError):
    """Raised when XML validation fails when saving."""
    pass
