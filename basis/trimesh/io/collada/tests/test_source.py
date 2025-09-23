# 这段代码通过单元测试验证了 COLLADA 模块中与数据源相关的基本功能,包括浮点源、ID 引用源和名称源的创建、保存和加载.
# 通过这些测试,可以确保在对 COLLADA 文件进行读写操作时,相关数据能够正确地保存和恢复.


import numpy
import collada
from collada.util import unittest
from collada.xmlutil import etree

fromstring = etree.fromstring
tostring = etree.tostring


class TestSource(unittest.TestCase):
    """
    测试Collada库中的Source模块,包括FloatSource、IDRefSource和NameSource的保存和加载功能.

    :param self: 测试类的实例
    """
    def setUp(self):
        """
        初始化测试环境,创建一个虚拟的Collada对象
        """
        self.dummy = collada.Collada(validate_output=True)

    def test_float_source_saving(self):
        """
        测试FloatSource对象的保存和加载功能
        """
        # 创建FloatSource对象
        floatsource = collada.source.FloatSource("myfloatsource", numpy.array([0.1, 0.2, 0.3]), ('X', 'Y', 'X'))
        self.assertEqual(floatsource.id, "myfloatsource")
        self.assertEqual(len(floatsource), 1)
        self.assertTupleEqual(floatsource.components, ('X', 'Y', 'X'))
        self.assertIsNotNone(str(floatsource))

        # 修改FloatSource对象的属性
        floatsource.id = "yourfloatsource"
        floatsource.components = ('S', 'T')
        floatsource.data = numpy.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        floatsource.save()

        # 加载并验证FloatSource对象
        loaded_floatsource = collada.source.Source.load(self.dummy, {}, fromstring(tostring(floatsource.xmlnode)))
        self.assertTrue(isinstance(loaded_floatsource, collada.source.FloatSource))
        self.assertEqual(floatsource.id, "yourfloatsource")
        self.assertEqual(len(floatsource), 3)
        self.assertTupleEqual(floatsource.components, ('S', 'T'))

    def test_idref_source_saving(self):
        """
        测试IDRefSource对象的保存和加载功能
        """
        # 创建IDRefSource对象
        idrefsource = collada.source.IDRefSource("myidrefsource",
                                                 numpy.array(['Ref1', 'Ref2'], dtype=numpy.string_),
                                                 ('MORPH_TARGET',))
        self.assertEqual(idrefsource.id, "myidrefsource")
        self.assertEqual(len(idrefsource), 2)
        self.assertTupleEqual(idrefsource.components, ('MORPH_TARGET',))
        self.assertIsNotNone(str(idrefsource))

        # 修改IDRefSource对象的属性
        idrefsource.id = "youridrefsource"
        idrefsource.components = ('JOINT_TARGET', 'WHATEVER_TARGET')
        idrefsource.data = numpy.array(['Ref5', 'Ref6', 'Ref7', 'Ref8', 'Ref9', 'Ref10'], dtype=numpy.string_)
        idrefsource.save()

        # 加载并验证IDRefSource对象
        loaded_idrefsource = collada.source.Source.load(self.dummy, {}, fromstring(tostring(idrefsource.xmlnode)))
        self.assertTrue(isinstance(loaded_idrefsource, collada.source.IDRefSource))
        self.assertEqual(loaded_idrefsource.id, "youridrefsource")
        self.assertEqual(len(loaded_idrefsource), 3)
        self.assertTupleEqual(loaded_idrefsource.components, ('JOINT_TARGET', 'WHATEVER_TARGET'))

    def test_name_source_saving(self):
        """
        测试NameSource对象的保存和加载功能
        """
        # 创建NameSource对象
        namesource = collada.source.NameSource("mynamesource",
                                               numpy.array(['Name1', 'Name2'], dtype=numpy.string_),
                                               ('JOINT',))
        self.assertEqual(namesource.id, "mynamesource")
        self.assertEqual(len(namesource), 2)
        self.assertTupleEqual(namesource.components, ('JOINT',))
        self.assertIsNotNone(str(namesource))

        # 修改NameSource对象的属性
        namesource.id = "yournamesource"
        namesource.components = ('WEIGHT', 'WHATEVER')
        namesource.data = numpy.array(['Name1', 'Name2', 'Name3', 'Name4', 'Name5', 'Name6'], dtype=numpy.string_)
        namesource.save()

        # 加载并验证NameSource对象
        loaded_namesource = collada.source.Source.load(self.dummy, {}, fromstring(tostring(namesource.xmlnode)))
        self.assertTrue(isinstance(loaded_namesource, collada.source.NameSource))
        self.assertEqual(loaded_namesource.id, "yournamesource")
        self.assertEqual(len(loaded_namesource), 3)
        self.assertTupleEqual(loaded_namesource.components, ('WEIGHT', 'WHATEVER'))


if __name__ == '__main__':
    unittest.main()
