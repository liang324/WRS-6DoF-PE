# 包含用于表示摄像机的对象

from .common import DaeIncompleteError
from .common import DaeMalformedError
from .common import DaeObject
from .common import DaeUnsupportedError
from .common import E


class Camera(DaeObject):
    """
    保存来自 <camera> 标签的数据的基础相机类
    """
    @staticmethod
    def load(collada, localscope, node):
        """
        从 XML 节点加载相机

        :param collada: COLLADA 对象实例
        :param localscope: 本地作用域
        :param node: XML 节点
        :return: PerspectiveCamera 或 OrthographicCamera 实例
        :raises DaeIncompleteError: 如果缺少常用技术
        :raises DaeUnsupportedError: 如果相机类型无法识别
        """
        tecnode = node.find('%s/%s' % (collada.tag('optics'), collada.tag('technique_common')))
        if tecnode is None or len(tecnode) == 0:
            raise DaeIncompleteError('Missing common technique in camera')
        camnode = tecnode[0]
        if camnode.tag == collada.tag('perspective'):
            return PerspectiveCamera.load(collada, localscope, node)
        elif camnode.tag == collada.tag('orthographic'):
            return OrthographicCamera.load(collada, localscope, node)
        else:
            raise DaeUnsupportedError('Unrecognized camera type: %s' % camnode.tag)


class PerspectiveCamera(Camera):
    """
    在 COLLADA 标签 <perspective> 中定义的透视相机
    """
    def __init__(self, id, znear, zfar, xfov=None, yfov=None,
                 aspect_ratio=None, xmlnode=None):
        """
        创建一个新的透视相机

        注意: ``aspect_ratio = tan(0.5*xfov) / tan(0.5*yfov)``

        可以指定以下之一:
         * 仅 :attr:`xfov`
         * 仅 :attr:`yfov`
         * :attr:`xfov` 和 :attr:`yfov`
         * :attr:`xfov` 和 :attr:`aspect_ratio`
         * :attr:`yfov` 和 :attr:`aspect_ratio`

        任何其他组合将引发 :class:`collada.common.DaeMalformedError`

        :param str id: 相机的标识符
        :param float znear: 到近剪裁面的距离
        :param float zfar: 到远剪裁面的距离
        :param float xfov: 水平视野,单位为度
        :param float yfov: 垂直视野,单位为度
        :param float aspect_ratio: 视野的纵横比
        :param xmlnode: 如果从 XML 加载,则为 XML 节点
        """
        self.id = id
        """Identifier for the camera"""
        self.xfov = xfov
        """Horizontal field of view, in degrees"""
        self.yfov = yfov
        """Vertical field of view, in degrees"""
        self.aspect_ratio = aspect_ratio
        """Aspect ratio of the field of view"""
        self.znear = znear
        """Distance to the near clipping plane"""
        self.zfar = zfar
        """Distance to the far clipping plane"""

        self._checkValidParams()

        if xmlnode is not None:
            self.xmlnode = xmlnode
            """ElementTree representation of the data."""
        else:
            self._recreateXmlNode()

    def _recreateXmlNode(self):
        perspective_node = E.perspective()
        if self.xfov is not None:
            perspective_node.append(E.xfov(str(self.xfov)))
        if self.yfov is not None:
            perspective_node.append(E.yfov(str(self.yfov)))
        if self.aspect_ratio is not None:
            perspective_node.append(E.aspect_ratio(str(self.aspect_ratio)))
        perspective_node.append(E.znear(str(self.znear)))
        perspective_node.append(E.zfar(str(self.zfar)))
        self.xmlnode = E.camera(
            E.optics(
                E.technique_common(perspective_node)
            )
            , id=self.id, name=self.id)

    def _checkValidParams(self):
        if self.xfov is not None and self.yfov is None \
                and self.aspect_ratio is None:
            pass
        elif self.xfov is None and self.yfov is not None \
                and self.aspect_ratio is None:
            pass
        elif self.xfov is not None and self.yfov is None \
                and self.aspect_ratio is not None:
            pass
        elif self.xfov is None and self.yfov is not None \
                and self.aspect_ratio is not None:
            pass
        elif self.xfov is not None and self.yfov is not None \
                and self.aspect_ratio is None:
            pass
        else:
            raise DaeMalformedError("Received invalid combination of xfov (%s), yfov (%s), and aspect_ratio (%s)" %
                                    (str(self.xfov), str(self.yfov), str(self.aspect_ratio)))

    def save(self):
        """Saves the perspective camera's properties back to xmlnode"""
        self._checkValidParams()
        self._recreateXmlNode()

    @staticmethod
    def load(collada, localscope, node):
        persnode = node.find('%s/%s/%s' % (collada.tag('optics'), collada.tag('technique_common'),
                                           collada.tag('perspective')))

        if persnode is None:
            raise DaeIncompleteError('Missing perspective for camera definition')

        xfov = persnode.find(collada.tag('xfov'))
        yfov = persnode.find(collada.tag('yfov'))
        aspect_ratio = persnode.find(collada.tag('aspect_ratio'))
        znearnode = persnode.find(collada.tag('znear'))
        zfarnode = persnode.find(collada.tag('zfar'))
        id = node.get('id', '')

        try:
            if xfov is not None:
                xfov = float(xfov.text)
            if yfov is not None:
                yfov = float(yfov.text)
            if aspect_ratio is not None:
                aspect_ratio = float(aspect_ratio.text)
            znear = float(znearnode.text)
            zfar = float(zfarnode.text)
        except (TypeError, ValueError) as ex:
            raise DaeMalformedError('Corrupted float values in camera definition')

        # There are some exporters that incorrectly output all three of these.
        # Worse, they actually got the calculation of aspect_ratio wrong!
        # So instead of failing to load, let's just add one more hack because of terrible exporters
        if xfov is not None and yfov is not None and aspect_ratio is not None:
            aspect_ratio = None

        return PerspectiveCamera(id, znear, zfar, xfov=xfov, yfov=yfov,
                                 aspect_ratio=aspect_ratio, xmlnode=node)

    def bind(self, matrix):
        """Create a bound camera of itself based on a transform matrix.

        :param numpy.array matrix:
          A numpy transformation matrix of size 4x4

        :rtype: :class:`collada.camera.BoundPerspectiveCamera`

        """
        return BoundPerspectiveCamera(self, matrix)

    def __str__(self):
        return '<PerspectiveCamera id=%s>' % self.id

    def __repr__(self):
        return str(self)


class OrthographicCamera(Camera):
    """
    在 COLLADA 标签 <orthographic> 中定义的正交相机.
    """
    def __init__(self, id, znear, zfar, xmag=None, ymag=None, aspect_ratio=None, xmlnode=None):
        """
        创建一个新的正交相机

        注意: ``aspect_ratio = xmag / ymag``

        可以指定以下之一:
         * 仅 :attr:`xmag`
         * 仅 :attr:`ymag`
         * :attr:`xmag` 和 :attr:`ymag`
         * :attr:`xmag` 和 :attr:`aspect_ratio`
         * :attr:`ymag` 和 :attr:`aspect_ratio`

        任何其他组合将引发 :class:`collada.common.DaeMalformedError`

        :param str id: 相机的标识符
        :param float znear: 到近剪裁面的距离
        :param float zfar: 到远剪裁面的距离
        :param float xmag: 视图的水平放大
        :param float ymag: 视图的垂直放大
        :param float aspect_ratio: 视野的纵横比
        :param xmlnode: 如果从 XML 加载,则为 XML 节点
        """
        self.id = id
        """Identifier for the camera"""
        self.xmag = xmag
        """Horizontal magnification of the view"""
        self.ymag = ymag
        """Vertical magnification of the view"""
        self.aspect_ratio = aspect_ratio
        """Aspect ratio of the field of view"""
        self.znear = znear
        """Distance to the near clipping plane"""
        self.zfar = zfar
        """Distance to the far clipping plane"""

        self._checkValidParams()

        if xmlnode is not None:
            self.xmlnode = xmlnode
            """ElementTree representation of the data."""
        else:
            self._recreateXmlNode()

    def _recreateXmlNode(self):
        orthographic_node = E.orthographic()
        if self.xmag is not None:
            orthographic_node.append(E.xmag(str(self.xmag)))
        if self.ymag is not None:
            orthographic_node.append(E.ymag(str(self.ymag)))
        if self.aspect_ratio is not None:
            orthographic_node.append(E.aspect_ratio(str(self.aspect_ratio)))
        orthographic_node.append(E.znear(str(self.znear)))
        orthographic_node.append(E.zfar(str(self.zfar)))
        self.xmlnode = E.camera(
            E.optics(
                E.technique_common(orthographic_node)
            )
            , id=self.id, name=self.id)

    def _checkValidParams(self):
        if self.xmag is not None and self.ymag is None \
                and self.aspect_ratio is None:
            pass
        elif self.xmag is None and self.ymag is not None \
                and self.aspect_ratio is None:
            pass
        elif self.xmag is not None and self.ymag is None \
                and self.aspect_ratio is not None:
            pass
        elif self.xmag is None and self.ymag is not None \
                and self.aspect_ratio is not None:
            pass
        elif self.xmag is not None and self.ymag is not None \
                and self.aspect_ratio is None:
            pass
        else:
            raise DaeMalformedError("Received invalid combination of xmag (%s), ymag (%s), and aspect_ratio (%s)" %
                                    (str(self.xmag), str(self.ymag), str(self.aspect_ratio)))

    def save(self):
        """Saves the orthographic camera's properties back to xmlnode"""
        self._checkValidParams()
        self._recreateXmlNode()

    @staticmethod
    def load(collada, localscope, node):
        orthonode = node.find('%s/%s/%s' % (
            collada.tag('optics'),
            collada.tag('technique_common'),
            collada.tag('orthographic')))

        if orthonode is None: raise DaeIncompleteError('Missing orthographic for camera definition')

        xmag = orthonode.find(collada.tag('xmag'))
        ymag = orthonode.find(collada.tag('ymag'))
        aspect_ratio = orthonode.find(collada.tag('aspect_ratio'))
        znearnode = orthonode.find(collada.tag('znear'))
        zfarnode = orthonode.find(collada.tag('zfar'))
        id = node.get('id', '')

        try:
            if xmag is not None:
                xmag = float(xmag.text)
            if ymag is not None:
                ymag = float(ymag.text)
            if aspect_ratio is not None:
                aspect_ratio = float(aspect_ratio.text)
            znear = float(znearnode.text)
            zfar = float(zfarnode.text)
        except (TypeError, ValueError) as ex:
            raise DaeMalformedError('Corrupted float values in camera definition')

        # There are some exporters that incorrectly output all three of these.
        # Worse, they actually got the calculation of aspect_ratio wrong!
        # So instead of failing to load, let's just add one more hack because of terrible exporters
        if xmag is not None and ymag is not None and aspect_ratio is not None:
            aspect_ratio = None

        return OrthographicCamera(id, znear, zfar, xmag=xmag, ymag=ymag,
                                  aspect_ratio=aspect_ratio, xmlnode=node)

    def bind(self, matrix):
        """Create a bound camera of itself based on a transform matrix.

        :param numpy.array matrix:
          A numpy transformation matrix of size 4x4

        :rtype: :class:`collada.camera.BoundOrthographicCamera`

        """
        return BoundOrthographicCamera(self, matrix)

    def __str__(self):
        return '<OrthographicCamera id=%s>' % self.id

    def __repr__(self):
        return str(self)


class BoundCamera(object):
    """
    Base class for bound cameras
    """
    pass


class BoundPerspectiveCamera(BoundCamera):
    """
    透视相机绑定到场景并进行变换.当相机在场景中实例化时创建.请勿手动创建此对象
    """

    def __init__(self, cam, matrix):
        """
        创建一个绑定的透视相机

        :param cam: 原始透视相机对象
        :param numpy.array matrix: 大小为 4x4 的 numpy 变换矩阵
        """
        self.xfov = cam.xfov
        """Horizontal field of view, in degrees"""
        self.yfov = cam.yfov
        """Vertical field of view, in degrees"""
        self.aspect_ratio = cam.aspect_ratio
        """Aspect ratio of the field of view"""
        self.znear = cam.znear
        """Distance to the near clipping plane"""
        self.zfar = cam.zfar
        """Distance to the far clipping plane"""
        self.matrix = matrix
        """The matrix bound to"""
        self.position = matrix[:3, 3]
        """The position of the camera"""
        self.direction = -matrix[:3, 2]
        """The direction the camera is facing"""
        self.up = matrix[:3, 1]
        """The up vector of the camera"""
        self.original = cam
        """Original :class:`collada.camera.PerspectiveCamera` objects this is bound to."""

    def __str__(self):
        return '<BoundPerspectiveCamera bound to %s>' % self.original.id

    def __repr__(self):
        return str(self)


class BoundOrthographicCamera(BoundCamera):
    """
    正交相机绑定到场景并进行变换.当相机在场景中实例化时创建.请勿手动创建此对象
    """

    def __init__(self, cam, matrix):
        self.xmag = cam.xmag
        """Horizontal magnification of the view"""
        self.ymag = cam.ymag
        """Vertical magnification of the view"""
        self.aspect_ratio = cam.aspect_ratio
        """Aspect ratio of the field of view"""
        self.znear = cam.znear
        """Distance to the near clipping plane"""
        self.zfar = cam.zfar
        """Distance to the far clipping plane"""
        self.matrix = matrix
        """The matrix bound to"""
        self.position = matrix[:3, 3]
        """The position of the camera"""
        self.direction = -matrix[:3, 2]
        """The direction the camera is facing"""
        self.up = matrix[:3, 1]
        """The up vector of the camera"""
        self.original = cam
        """Original :class:`collada.camera.OrthographicCamera` objects this is bound to."""

    def __str__(self):
        return '<BoundOrthographicCamera bound to %s>' % self.original.id

    def __repr__(self):
        return str(self)
