# 这段代码是一个 Python 单元测试,用于测试 basis.trimesh.io.collada 模块中与 COLLADA 文件 asset 相关的类
# 包括 collada.asset.Contributor 和 collada.asset.Asset 类.
# COLLADA 文件中的 asset 元素用于存储文件的元数据信息,例如作者、创建时间、修改时间、单位等

import datetime
from fontTools.misc import etree
from basis.trimesh.io import collada
from basis.trimesh.io.collada.util import unittest

fromstring = etree.fromstring
tostring = etree.tostring


class TestAsset(unittest.TestCase):
    """
    测试 Collada库中的资产模块

    :param self: 测试类的实例
    """
    def setUp(self):
        """
        在每个测试方法之前运行的设置方法
        初始化一个Collada对象,用于测试

        :param self: 测试类的实例
        """
        self.dummy = collada.Collada(validate_output=True)

    def test_asset_contributor(self):
        """
        测试Contributor类的功能

        :param self: 测试类的实例
        """
        contributor = collada.asset.Contributor()
        # 验证贡献者的初始属性是否为 None
        self.assertIsNone(contributor.author)  # 验证作者属性
        self.assertIsNone(contributor.authoring_tool)  # 验证作者工具属性
        self.assertIsNone(contributor.comments)  # 验证评论属性
        self.assertIsNone(contributor.copyright)  # 验证版权属性
        self.assertIsNone(contributor.source_data)  # 验证源数据属性

        # 保存贡献者并加载
        contributor.save()
        contributor = collada.asset.Contributor.load(self.dummy, {}, fromstring(tostring(contributor.xmlnode)))
        self.assertIsNone(contributor.author)
        self.assertIsNone(contributor.authoring_tool)
        self.assertIsNone(contributor.comments)
        self.assertIsNone(contributor.copyright)
        self.assertIsNone(contributor.source_data)

        # 设置贡献者的属性
        contributor.author = "author1"
        contributor.authoring_tool = "tool2"
        contributor.comments = "comments3"
        contributor.copyright = "copyright4"
        contributor.source_data = "data5"

        # 保存并加载贡献者,验证属性
        contributor.save()
        contributor = collada.asset.Contributor.load(self.dummy, {}, fromstring(tostring(contributor.xmlnode)))
        self.assertEqual(contributor.author, "author1")
        self.assertEqual(contributor.authoring_tool, "tool2")
        self.assertEqual(contributor.comments, "comments3")
        self.assertEqual(contributor.copyright, "copyright4")
        self.assertEqual(contributor.source_data, "data5")

    def test_asset(self):
        """
        测试Asset类的功能

        :param self: 测试类的实例
        """
        asset = collada.asset.Asset()
        # 验证资产的初始属性是否为 None
        self.assertIsNone(asset.title)  # 验证标题属性
        self.assertIsNone(asset.subject)  # 验证主题属性
        self.assertIsNone(asset.revision)  # 验证修订属性
        self.assertIsNone(asset.keywords)  # 验证关键词属性
        self.assertIsNone(asset.unitname)  # 验证单位名称属性
        self.assertIsNone(asset.unitmeter)  # 验证单位米属性
        self.assertEqual(asset.contributors, [])  # 验证贡献者列表
        self.assertEqual(asset.upaxis, collada.asset.UP_AXIS.Y_UP)  # 验证上轴
        self.assertIsInstance(asset.created, datetime.datetime)  # 验证创建时间
        self.assertIsInstance(asset.modified, datetime.datetime)  # 验证修改时间

        # 保存并重新加载Asset对象,检查默认属性值是否保持不变
        asset.save()
        asset = collada.asset.Asset.load(self.dummy, {}, fromstring(tostring(asset.xmlnode)))

        # 验证加载后的资产属性
        self.assertIsNone(asset.title)
        self.assertIsNone(asset.subject)
        self.assertIsNone(asset.revision)
        self.assertIsNone(asset.keywords)
        self.assertIsNone(asset.unitname)
        self.assertIsNone(asset.unitmeter)
        self.assertEqual(asset.contributors, [])
        self.assertEqual(asset.upaxis, collada.asset.UP_AXIS.Y_UP)
        self.assertIsInstance(asset.created, datetime.datetime)
        self.assertIsInstance(asset.modified, datetime.datetime)

        # 设置各种属性并验证它们是否正确保存和重新加载
        asset.title = 'title1'
        asset.subject = 'subject2'
        asset.revision = 'revision3'
        asset.keywords = 'keywords4'
        asset.unitname = 'feet'
        asset.unitmeter = 3.1
        contrib1 = collada.asset.Contributor(author="jeff")
        contrib2 = collada.asset.Contributor(author="bob")
        asset.contributors = [contrib1, contrib2]
        asset.upaxis = collada.asset.UP_AXIS.Z_UP
        time1 = datetime.datetime.now()
        asset.created = time1
        time2 = datetime.datetime.now() + datetime.timedelta(hours=5)
        asset.modified = time2

        # 保存并加载资产,验证属性
        asset.save()
        asset = collada.asset.Asset.load(self.dummy, {}, fromstring(tostring(asset.xmlnode)))
        self.assertEqual(asset.title, 'title1')
        self.assertEqual(asset.subject, 'subject2')
        self.assertEqual(asset.revision, 'revision3')
        self.assertEqual(asset.keywords, 'keywords4')
        self.assertEqual(asset.unitname, 'feet')
        self.assertEqual(asset.unitmeter, 3.1)
        self.assertEqual(asset.upaxis, collada.asset.UP_AXIS.Z_UP)
        self.assertEqual(asset.created, time1)
        self.assertEqual(asset.modified, time2)
        self.assertEqual(len(asset.contributors), 2)


if __name__ == '__main__':
    unittest.main()
