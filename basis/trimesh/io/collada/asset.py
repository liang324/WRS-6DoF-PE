# 包含COLLADA资产信息

import numpy
import datetime
import dateutil.parser

from .common import DaeObject, E, tag
from .common import DaeIncompleteError, DaeBrokenRefError, \
    DaeMalformedError, DaeUnsupportedError
from .util import _correctValInNode
from .xmlutil import etree as ElementTree


class UP_AXIS:
    """The up-axis of the collada document."""
    X_UP = 'X_UP'
    """Indicates X direction is up"""
    Y_UP = 'Y_UP'
    """Indicates Y direction is up"""
    Z_UP = 'Z_UP'
    """Indicates Z direction is up"""


class Contributor(DaeObject):
    """
    定义用于资产管理的作者信息
    """

    def __init__(self, author=None, authoring_tool=None, comments=None, copyright=None, source_data=None, xmlnode=None):
        """
        创建一个新的 贡献者 contributor

        :param str author: 作者的名字
        :param str authoring_tool: 作者工具的名称
        :param str comments: 贡献者的评论
        :param str copyright: 版权信息
        :param str source_data: 引用源数据的 URI
        :param xmlnode: 如果是从 XML 加载的,则为 XML 节点
        """
        self.author = author  # 包含作者名字的字符串
        self.authoring_tool = authoring_tool  # 包含作者工具名称的字符串
        self.comments = comments  # 包含贡献者评论的字符串
        self.copyright = copyright  # 包含版权信息的字符串
        self.source_data = source_data  # 包含引用此资产源数据的 URI 的字符串

        if xmlnode is not None:
            self.xmlnode = xmlnode  # 贡献者的 ElementTree 表示

        else:
            self.xmlnode = E.contributor()
            if author is not None:
                self.xmlnode.append(E.author(str(author)))
            if authoring_tool is not None:
                self.xmlnode.append(E.authoring_tool(str(authoring_tool)))
            if comments is not None:
                self.xmlnode.append(E.comments(str(comments)))
            if copyright is not None:
                self.xmlnode.append(E.copyright(str(copyright)))
            if source_data is not None:
                self.xmlnode.append(E.source_data(str(source_data)))

    @staticmethod
    def load(collada, localscope, node):
        author = node.find(collada.tag('author'))
        authoring_tool = node.find(collada.tag('authoring_tool'))
        comments = node.find(collada.tag('comments'))
        copyright = node.find(collada.tag('copyright'))
        source_data = node.find(collada.tag('source_data'))
        if author is not None: author = author.text
        if authoring_tool is not None: authoring_tool = authoring_tool.text
        if comments is not None: comments = comments.text
        if copyright is not None: copyright = copyright.text
        if source_data is not None: source_data = source_data.text
        return Contributor(author=author, authoring_tool=authoring_tool,
                           comments=comments, copyright=copyright, source_data=source_data, xmlnode=node)

    def save(self):
        """
        将贡献者信息保存回 :attr:`xmlnode
        """
        _correctValInNode(self.xmlnode, 'author', self.author)
        _correctValInNode(self.xmlnode, 'authoring_tool', self.authoring_tool)
        _correctValInNode(self.xmlnode, 'comments', self.comments)
        _correctValInNode(self.xmlnode, 'copyright', self.copyright)
        _correctValInNode(self.xmlnode, 'source_data', self.source_data)

    def __str__(self):
        return '<Contributor author=%s>' % (str(self.author),)

    def __repr__(self):
        return str(self)


class Asset(DaeObject):
    """
    定义资产管理信息
    """
    def __init__(self, created=None, modified=None, title=None, subject=None, revision=None,
                 keywords=None, unitname=None, unitmeter=None, upaxis=None, contributors=None, xmlnode=None):
        """
        创建一组新的资产信息

        :param datetime.datetime created: 资产创建的时间.如果为 None,则设置为当前日期和时间.
        :param datetime.datetime modified: 资产修改的时间.如果为 None,则设置为当前日期和时间.
        :param str title: 资产的标题
        :param str subject: 资产的主题描述
        :param str revision: 关于资产的修订信息
        :param str keywords: 用于资产搜索的关键词列表
        :param str unitname: 此资产的距离单位名称
        :param float unitmeter: 一个距离单位中有多少真实世界的米
        :param `collada.asset.UP_AXIS` upaxis: 资产的上轴.如果为 None,则设置为 Y_UP
        :param list contributors: 资产的贡献者列表
        :param xmlnode: 如果是从 XML 加载的,则为 XML 节点
        """
        if created is None:
            created = datetime.datetime.now()
        self.created = created
        """Instance of :class:`datetime.datetime` indicating when the asset was created"""

        if modified is None:
            modified = datetime.datetime.now()
        self.modified = modified
        """Instance of :class:`datetime.datetime` indicating when the asset was modified"""

        self.title = title
        """String containing the title of the asset"""
        self.subject = subject
        """String containing the description of the topical subject of the asset"""
        self.revision = revision
        """String containing revision information about the asset"""
        self.keywords = keywords
        """String containing a list of words used for search criteria for the asset"""
        self.unitname = unitname
        """String containing the name of the unit of distance for this asset"""
        self.unitmeter = unitmeter
        """Float containing how many real-world meters are in one distance unit"""

        if upaxis is None:
            upaxis = UP_AXIS.Y_UP
        self.upaxis = upaxis
        """Instance of type :class:`collada.asset.UP_AXIS` indicating the up-axis of the asset"""

        if contributors is None:
            contributors = []
        self.contributors = contributors
        """A list of instances of :class:`collada.asset.Contributor`"""

        if xmlnode is not None:
            self.xmlnode = xmlnode
            """ElementTree representation of the asset."""
        else:
            self._recreateXmlNode()

    def _recreateXmlNode(self):
        self.xmlnode = E.asset()
        for contributor in self.contributors:
            self.xmlnode.append(contributor.xmlnode)
        self.xmlnode.append(E.created(self.created.isoformat()))
        if self.keywords is not None:
            self.xmlnode.append(E.keywords(self.keywords))
        self.xmlnode.append(E.modified(self.modified.isoformat()))
        if self.revision is not None:
            self.xmlnode.append(E.revision(self.revision))
        if self.subject is not None:
            self.xmlnode.append(E.subject(self.subject))
        if self.title is not None:
            self.xmlnode.append(E.title(self.title))
        if self.unitmeter is not None and self.unitname is not None:
            self.xmlnode.append(E.unit(name=self.unitname, meter=str(self.unitmeter)))
        self.xmlnode.append(E.up_axis(self.upaxis))

    def save(self):
        """Saves the asset info back to :attr:`xmlnode`"""
        self._recreateXmlNode()

    @staticmethod
    def load(collada, localscope, node):
        contributornodes = node.findall(collada.tag('contributor'))
        contributors = []
        for contributornode in contributornodes:
            contributors.append(Contributor.load(collada, localscope, contributornode))

        created = node.find(collada.tag('created'))
        if created is not None:
            try:
                created = dateutil.parser.parse(created.text)
            except:
                created = None

        keywords = node.find(collada.tag('keywords'))
        if keywords is not None: keywords = keywords.text

        modified = node.find(collada.tag('modified'))
        if modified is not None:
            try:
                modified = dateutil.parser.parse(modified.text)
            except:
                modified = None

        revision = node.find(collada.tag('revision'))
        if revision is not None: revision = revision.text

        subject = node.find(collada.tag('subject'))
        if subject is not None: subject = subject.text

        title = node.find(collada.tag('title'))
        if title is not None: title = title.text

        unitnode = node.find(collada.tag('unit'))
        if unitnode is not None:
            unitname = unitnode.get('name')
            try:
                unitmeter = float(unitnode.get('meter'))
            except:
                unitname = None
                unitmeter = None
        else:
            unitname = None
            unitmeter = None

        upaxis = node.find(collada.tag('up_axis'))
        if upaxis is not None:
            upaxis = upaxis.text
            if not (upaxis == UP_AXIS.X_UP or upaxis == UP_AXIS.Y_UP or \
                    upaxis == UP_AXIS.Z_UP):
                upaxis = None

        return Asset(created=created, modified=modified, title=title,
                     subject=subject, revision=revision, keywords=keywords,
                     unitname=unitname, unitmeter=unitmeter, upaxis=upaxis,
                     contributors=contributors, xmlnode=node)

    def __str__(self):
        return '<Asset title=%s>' % (str(self.title),)

    def __repr__(self):
        return str(self)
