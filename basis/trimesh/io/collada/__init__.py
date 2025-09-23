# collada (pycollada)包的主模块,你会在这里找到`Collada`类
# 它是用来访问Collada文件的,如果输入文件不是预期的,则会引发一些异常


__version__ = "0.4.1"

import os.path
import posixpath
import traceback
import types
import zipfile
from datetime import datetime
from . import animation
from . import asset
from . import camera
from . import controller
from . import geometry
from . import light
from . import material
from . import scene
from .common import E, tagger, tag
from .common import DaeError, DaeObject, DaeIncompleteError, \
    DaeBrokenRefError, DaeMalformedError, DaeUnsupportedError, \
    DaeSaveValidationError
from .util import basestring, BytesIO
from .util import IndexedList
from .xmlutil import etree as ElementTree
from .xmlutil import writeXML

try:
    from . import schema
except ImportError:  # no lxml
    schema = None


class Collada(object):
    """
    用于创建和加载 COLLADA 文档的主要类
    """
    geometries = property(lambda s: s._geometries, lambda s, v: s._setIndexedList('_geometries', v), doc="""
    A list of :class:`collada.geometry.Geometry` objects. Can also be indexed by id""")
    controllers = property(lambda s: s._controllers, lambda s, v: s._setIndexedList('_controllers', v), doc="""
    A list of :class:`collada.controller.Controller` objects. Can also be indexed by id""")
    animations = property(lambda s: s._animations, lambda s, v: s._setIndexedList('_animations', v), doc="""
    A list of :class:`collada.animation.Animation` objects. Can also be indexed by id""")
    lights = property(lambda s: s._lights, lambda s, v: s._setIndexedList('_lights', v), doc="""
    A list of :class:`collada.light.Light` objects. Can also be indexed by id""")
    cameras = property(lambda s: s._cameras, lambda s, v: s._setIndexedList('_cameras', v), doc="""
    A list of :class:`collada.camera.Camera` objects. Can also be indexed by id""")
    images = property(lambda s: s._images, lambda s, v: s._setIndexedList('_images', v), doc="""
    A list of :class:`collada.material.CImage` objects. Can also be indexed by id""")
    effects = property(lambda s: s._effects, lambda s, v: s._setIndexedList('_effects', v), doc="""
    A list of :class:`collada.material.Effect` objects. Can also be indexed by id""")
    materials = property(lambda s: s._materials, lambda s, v: s._setIndexedList('_materials', v), doc="""
    A list of :class:`collada.material.Effect` objects. Can also be indexed by id""")
    nodes = property(lambda s: s._nodes, lambda s, v: s._setIndexedList('_nodes', v), doc="""
    A list of :class:`collada.scene.Node` objects. Can also be indexed by id""")
    scenes = property(lambda s: s._scenes, lambda s, v: s._setIndexedList('_scenes', v), doc="""
    A list of :class:`collada.scene.Scene` objects. Can also be indexed by id""")

    def __init__(self,
                 filename=None,
                 ignore=None,
                 aux_file_loader=None,
                 zip_filename=None,
                 validate_output=False):
        """
        初始化 Collada 实例

        :param filename: 包含要打开的文件路径的字符串或类似文件的对象.支持未压缩的 .dae 文件以及 zip 文件存档
                         如果设置为 `None`,则创建一个新的 COLLADA 实例
        :param list ignore: 应在加载 COLLADA 文档时忽略的 `common.DaeError` 类型列表
                            这些类型的实例将在加载后添加到 `errors` 属性中,但不会被抛出.仅在 `filename` 不为 `None` 时使用
        :param function aux_file_loader: 当从本地文件系统读取时,引用的文件(例如纹理图像)将从磁盘加载
                                         当从 zip 文件加载时,将从 zip 存档中加载.如果这些文件来自其他来源(例如数据库)并且/或者您使用 StringIO 加载
                                         请将此参数设置为一个函数,该函数接收一个文件名并返回文件中的二进制数据
                                         如果 `filename` 为 `None`,并且您希望加载辅助文件,则必须设置此参数
        :param str zip_filename: 如果加载的文件是 zip 存档,您可以设置此参数以指示应加载存档中的哪个文件
                                如果未设置,将搜索以 .dae 结尾的文件
        :param bool validate_output: 如果设置为 True,则在调用 `save` 方法时写入的 XML 将根据 COLLADA 1.4.1 模式进行验证
                                    如果验证失败,将抛出 `common.DaeSaveValidationError` 异常
        """
        self.errors = []
        """List of :class:`common.common.DaeError` objects representing errors encountered while loading collada file"""
        self.assetInfo = None
        """Instance of :class:`collada.asset.Asset` containing asset information"""

        self._geometries = IndexedList([], ('id',))
        self._controllers = IndexedList([], ('id',))
        self._animations = IndexedList([], ('id',))
        self._lights = IndexedList([], ('id',))
        self._cameras = IndexedList([], ('id',))
        self._images = IndexedList([], ('id',))
        self._effects = IndexedList([], ('id',))
        self._materials = IndexedList([], ('id',))
        self._nodes = IndexedList([], ('id',))
        self._scenes = IndexedList([], ('id',))

        self.scene = None
        """The default scene. This is either an instance of :class:`collada.scene.Scene` or `None`."""

        # a function which will apply the namespace
        self.tag = tag

        if validate_output and schema:
            self.validator = schema.ColladaValidator()
        else:
            self.validator = None

        self.maskedErrors = []
        if ignore is not None:
            self.ignoreErrors(*ignore)

        if filename is None:
            self.filename = None
            self.zfile = None
            self.getFileData = self._nullGetFile
            if aux_file_loader is not None:
                self.getFileData = self._wrappedFileLoader(aux_file_loader)

            self.xmlnode = ElementTree.ElementTree(
                E.COLLADA(
                    E.library_cameras(),
                    E.library_controllers(),
                    E.library_effects(),
                    E.library_geometries(),
                    E.library_images(),
                    E.library_lights(),
                    E.library_materials(),
                    E.library_nodes(),
                    E.library_visual_scenes(),
                    E.scene(),
                    version='1.4.1'))
            """ElementTree representation of the collada document"""

            self.assetInfo = asset.Asset()
            return

        strdata = None
        if isinstance(filename, basestring):
            with open(filename, 'rb') as f:
                strdata = f.read()
            self.filename = filename
            self.getFileData = self._getFileFromDisk
        else:
            strdata = filename.read()  # assume it is a file like objects
            self.filename = None
            self.getFileData = self._nullGetFile

        try:
            self.zfile = zipfile.ZipFile(BytesIO(strdata), 'r')
        except:
            self.zfile = None

        if self.zfile:
            self.filename = ''
            daefiles = []
            if zip_filename is not None:
                self.filename = zip_filename
            else:
                for name in self.zfile.namelist():
                    if name.upper().endswith('.DAE'):
                        daefiles.append(name)
                for name in daefiles:
                    if not self.filename:
                        self.filename = name
                    elif "MACOSX" in self.filename:
                        self.filename = name
            if not self.filename or self.filename not in self.zfile.namelist():
                raise DaeIncompleteError('COLLADA file not found inside zip compressed file')
            data = self.zfile.read(self.filename)
            self.getFileData = self._getFileFromZip
        else:
            data = strdata

        if aux_file_loader is not None:
            self.getFileData = self._wrappedFileLoader(aux_file_loader)

        etree_parser = ElementTree.XMLParser()
        try:
            self.xmlnode = ElementTree.ElementTree(element=None,
                                                   file=BytesIO(data))
        except ElementTree.ParseError as e:
            raise DaeMalformedError("XML Parsing Error: %s" % e)

        # if we can't get the current namespace
        # the tagger from above will use a hardcoded default
        try:
            # get the root node, same for both etree and lxml
            xml_root = self.xmlnode.getroot()
            if hasattr(xml_root, 'nsmap'):
                # lxml has an nsmap
                # use the first value in the namespace map
                namespace = next(iter(xml_root.nsmap.values()))
            elif hasattr(xml_root, 'tag'):
                # for xml.etree we need to extract ns from root tag
                namespace = xml_root.tag.split('}')[0].lstrip('{')
            # create a tagging function using the extracted namespace
            self.tag = tagger(namespace)
        except BaseException:
            # failed to extract a namespace, using default
            traceback.print_exc()

        # functions which will load various things into collada objects
        self._loadAssetInfo()
        self._loadImages()
        self._loadEffects()
        self._loadMaterials()
        self._loadAnimations()
        self._loadGeometry()
        self._loadControllers()
        self._loadLights()
        self._loadCameras()
        self._loadNodes()
        self._loadScenes()
        self._loadDefaultScene()

    def _setIndexedList(self, propname, data):
        setattr(self, propname, IndexedList(data, ('id',)))

    def handleError(self, error):
        self.errors.append(error)
        if not type(error) in self.maskedErrors:
            raise

    def ignoreErrors(self, *args):
        """Add exceptions to the mask for ignoring or clear the mask if None given.

        You call c.ignoreErrors(e1, e2, ... ) if you want the loader to ignore those
        exceptions and continue loading whatever it can. If you want to empty the
        mask so all exceptions abort the load just call c.ignoreErrors(None).

        """
        if args == [None]:
            self.maskedErrors = []
        else:
            for e in args: self.maskedErrors.append(e)

    def _getFileFromZip(self, fname):
        """
        从 zip 存档中返回辅助文件的二进制数据作为字符串

        :param fname: 辅助文件的文件名
        :return: 文件的二进制数据
        """
        if not self.zfile:
            raise DaeBrokenRefError('Trying to load an auxiliary file %s but we are not reading from a zip' % fname)
        basepath = posixpath.dirname(self.filename)
        aux_path = posixpath.normpath(posixpath.join(basepath, fname))
        if aux_path not in self.zfile.namelist():
            raise DaeBrokenRefError('Auxiliar file %s not found in archive' % fname)
        return self.zfile.read(aux_path)

    def _getFileFromDisk(self, fname):
        """
        从本地磁盘返回辅助文件的二进制数据,相对于加载的文件路径

        :param fname: 辅助文件的文件名
        :return: 文件的二进制数据
        """
        if self.zfile:
            raise DaeBrokenRefError(
                'Trying to load an auxiliary file %s from disk but we are reading from a zip file' % fname)
        basepath = os.path.dirname(self.filename)
        aux_path = os.path.normpath(os.path.join(basepath, fname))
        if not os.path.exists(aux_path):
            raise DaeBrokenRefError('Auxiliar file %s not found on disk' % fname)
        fdata = open(aux_path, 'rb')
        return fdata.read()

    def _wrappedFileLoader(self, aux_file_loader):
        """
        包装辅助文件加载器以处理文件未找到的情况

        :param aux_file_loader: 用于加载辅助文件的函数
        :return: 包装后的文件加载函数
        """
        def __wrapped(fname):
            res = aux_file_loader(fname)
            if res is None:
                raise DaeBrokenRefError('Auxiliar file %s from auxiliary file loader not found' % fname)
            return res

        return __wrapped

    def _nullGetFile(self, fname):
        """
        处理尝试加载辅助文件但未从磁盘、zip或自定义处理程序加载的情况

        :param fname: 辅助文件的文件名
        :raise DaeBrokenRefError: 当无法加载辅助文件时抛出异常
        """
        raise DaeBrokenRefError(
            'Trying to load auxiliary file but collada was not loaded from disk, zip, or with custom handler')

    def _loadAssetInfo(self):
        """
        加载 <asset> 标签中的信息
        """
        assetnode = self.xmlnode.find(self.tag('asset'))
        if assetnode is not None:
            self.assetInfo = asset.Asset.load(self, {}, assetnode)
        else:
            self.assetInfo = asset.Asset()

    def _loadGeometry(self):
        """
        加载几何库
        """
        libnodes = self.xmlnode.findall(self.tag('library_geometries'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for geomnode in libnode.findall(self.tag('geometry')):
                        if geomnode.find(self.tag('mesh')) is None:
                            continue
                        try:
                            G = geometry.Geometry.load(self, {}, geomnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.geometries.append(G)

    def _loadControllers(self):
        """
        加载控制器库
        """
        libnodes = self.xmlnode.findall(self.tag('library_controllers'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for controlnode in libnode.findall(self.tag('controller')):
                        if controlnode.find(self.tag('skin')) is None \
                                and controlnode.find(self.tag('morph')) is None:
                            continue
                        try:
                            C = controller.Controller.load(self, {}, controlnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.controllers.append(C)

    def _loadAnimations(self):
        """
        加载动画库
        """
        libnodes = self.xmlnode.findall(self.tag('library_animations'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for animnode in libnode.findall(self.tag('animation')):
                        try:
                            A = animation.Animation.load(self, {}, animnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.animations.append(A)

    def _loadLights(self):
        """
        加载灯光库
        """
        libnodes = self.xmlnode.findall(self.tag('library_lights'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for lightnode in libnode.findall(self.tag('light')):
                        try:
                            lig = light.Light.load(self, {}, lightnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.lights.append(lig)

    def _loadCameras(self):
        """
        加载摄像机库
        """
        libnodes = self.xmlnode.findall(self.tag('library_cameras'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for cameranode in libnode.findall(self.tag('camera')):
                        try:
                            cam = camera.Camera.load(self, {}, cameranode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.cameras.append(cam)

    def _loadImages(self):
        """
        加载图像库
        """
        libnodes = self.xmlnode.findall(self.tag('library_images'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for imgnode in libnode.findall(self.tag('image')):
                        try:
                            img = material.CImage.load(self, {}, imgnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.images.append(img)

    def _loadEffects(self):
        """
        加载效果库
        """
        libnodes = self.xmlnode.findall(self.tag('library_effects'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for effectnode in libnode.findall(self.tag('effect')):
                        try:
                            effect = material.Effect.load(self, {}, effectnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.effects.append(effect)

    def _loadMaterials(self):
        """
        加载材质库
        """
        libnodes = self.xmlnode.findall(self.tag('library_materials'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for materialnode in libnode.findall(self.tag('material')):
                        try:
                            mat = material.Material.load(self, {}, materialnode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.materials.append(mat)

    def _loadNodes(self):
        """
        加载节点库
        """
        libnodes = self.xmlnode.findall(self.tag('library_nodes'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    tried_loading = []
                    succeeded = False
                    for node in libnode.findall(self.tag('node')):
                        try:
                            N = scene.loadNode(self, node, {})
                        except scene.DaeInstanceNotLoadedError as ex:
                            tried_loading.append((node, ex))
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            if N is not None:
                                self.nodes.append(N)
                                succeeded = True
                    while len(tried_loading) > 0 and succeeded:
                        succeeded = False
                        next_tried = []
                        for node, ex in tried_loading:
                            try:
                                N = scene.loadNode(self, node, {})
                            except scene.DaeInstanceNotLoadedError as ex:
                                next_tried.append((node, ex))
                            except DaeError as ex:
                                self.handleError(ex)
                            else:
                                if N is not None:
                                    self.nodes.append(N)
                                    succeeded = True
                        tried_loading = next_tried
                    if len(tried_loading) > 0:
                        for node, ex in tried_loading:
                            raise DaeBrokenRefError(ex.msg)

    def _loadScenes(self):
        """
        加载场景库
        """
        libnodes = self.xmlnode.findall(self.tag('library_visual_scenes'))
        if libnodes is not None:
            for libnode in libnodes:
                if libnode is not None:
                    for scenenode in libnode.findall(self.tag('visual_scene')):
                        try:
                            S = scene.Scene.load(self, scenenode)
                        except DaeError as ex:
                            self.handleError(ex)
                        else:
                            self.scenes.append(S)

    def _loadDefaultScene(self):
        """
        从根节点的 <scene> 标签加载默认场景
        """
        node = self.xmlnode.find('%s/%s' % (self.tag('scene'), self.tag('instance_visual_scene')))
        try:
            if node != None:
                sceneid = node.get('url')
                if not sceneid.startswith('#'):
                    raise DaeMalformedError('Malformed default scene reference to %s: ' % sceneid)
                self.scene = self.scenes.get(sceneid[1:])
                if not self.scene:
                    raise DaeBrokenRefError('Default scene %s not found' % sceneid)
        except DaeError as ex:
            self.handleError(ex)

    def save(self):
        """
        将 COLLADA 文档保存回 :attr:`xmlnode
        """
        libraries = [(self.geometries, 'library_geometries'),
                     (self.controllers, 'library_controllers'),
                     (self.lights, 'library_lights'),
                     (self.cameras, 'library_cameras'),
                     (self.images, 'library_images'),
                     (self.effects, 'library_effects'),
                     (self.materials, 'library_materials'),
                     (self.nodes, 'library_nodes'),
                     (self.scenes, 'library_visual_scenes')]

        self.assetInfo.save()
        assetnode = self.xmlnode.getroot().find(self.tag('asset'))
        if assetnode is not None:
            self.xmlnode.getroot().remove(assetnode)
        self.xmlnode.getroot().insert(0, self.assetInfo.xmlnode)

        library_loc = 0
        for i, node in enumerate(self.xmlnode.getroot()):
            if node.tag == self.tag('asset'):
                library_loc = i + 1

        for arr, name in libraries:
            node = self.xmlnode.find(self.tag(name))
            if node is None:
                if len(arr) == 0:
                    continue
                self.xmlnode.getroot().insert(library_loc, E(name))
                node = self.xmlnode.find(self.tag(name))
            elif node is not None and len(arr) == 0:
                self.xmlnode.getroot().remove(node)
                continue

            for o in arr:
                o.save()
                if o.xmlnode not in node:
                    node.append(o.xmlnode)
            xmlnodes = [o.xmlnode for o in arr]
            for n in node:
                if n not in xmlnodes:
                    node.remove(n)

        scenenode = self.xmlnode.find(self.tag('scene'))
        scenenode.clear()
        if self.scene is not None:
            sceneid = self.scene.id
            if sceneid not in self.scenes:
                raise DaeBrokenRefError('Default scene %s not found' % sceneid)
            scenenode.append(E.instance_visual_scene(url="#%s" % sceneid))

        if self.validator is not None:
            if not self.validator.validate(self.xmlnode):
                raise DaeSaveValidationError("Validation error when saving: " +
                                             self.validator.COLLADA_SCHEMA_1_4_1_INSTANCE.error_log.last_error.message)

    def write(self, fp):
        """
        将 COLLADA 文档写入文件.注意这也会调用 :meth:`save`,因此避免调用这两个方法以节省性能.

        :param file:
        文件名或类似文件的对象
        """
        self.save()
        if isinstance(fp, basestring):
            fp = open(fp, 'wb')
        writeXML(self.xmlnode, fp)

    def __str__(self):
        return '<Collada geometries=%d>' % (len(self.geometries))

    def __repr__(self):
        return str(self)
