# 这段代码通过单元测试验证了 COLLADA 模块中与线集相关的基本功能,包括线集的构造、保存和加载
# 通过这些测试,可以确保在对 COLLADA 文件进行读写操作时,线集的相关数据能够正确地保存和恢复



import numpy
from numpy.testing import assert_array_equal
import collada
from collada.util import unittest
from collada.xmlutil import etree
from collada.common import DaeIncompleteError

fromstring = etree.fromstring
tostring = etree.tostring


class TestLineset(unittest.TestCase):
    """
    测试Collada库中的LineSet模块

    :param self: 测试类的实例
    """
    def setUp(self):
        """
        初始化测试环境
        """
        self.dummy = collada.Collada(validate_output=True)

    def test_lineset_construction(self):
        """
        测试LineSet对象的构造

        - 空的sources字典应抛出异常
        - 如果没有定义VERTEX作为键,也应抛出异常
        - 添加一个定义了VERTEX但为空的输入列表应抛出错误
        """
        # 空的sources字典应抛出异常
        with self.assertRaises(DaeIncompleteError) as e:
            collada.lineset.LineSet({}, None, None)
        self.assertIn("at least one", e.exception.msg)

        # 如果没有定义VERTEX作为键,也应抛出异常
        with self.assertRaises(DaeIncompleteError) as e:
            collada.lineset.LineSet({"a": []}, None, None)
        self.assertIn("requires vertex", e.exception.msg)

        # 添加一个定义了VERTEX但为空的输入列表应抛出错误
        with self.assertRaises(DaeIncompleteError) as e:
            collada.lineset.LineSet({'VERTEX': []}, None, None)
        self.assertIn("requires vertex", e.exception.msg)

    def test_empty_lineset_saving(self):
        """
        测试空LineSet对象的保存和加载

        - 创建一个LineSet对象并验证其初始属性
        - 序列化和反序列化后,验证反序列化对象的属性与原始对象一致
        """
        linefloats = [1, 1, -1, 1, -1, -1, -1, -0.9999998, -1, -0.9999997, 1, -1, 1, 0.9999995, 1, 0.9999994, -1.000001,
                      1]
        linefloatsrc = collada.source.FloatSource("mylinevertsource", numpy.array(linefloats), ('X', 'Y', 'Z'))
        geometry = collada.geometry.Geometry(self.dummy, "geometry0", "mygeometry", [linefloatsrc])
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', "#mylinevertsource")
        indices = numpy.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        lineset = geometry.createLineSet(indices, input_list, "mymaterial")

        # 检查LineSet对象的初始值
        self.assertIsNotNone(str(lineset))
        assert_array_equal(lineset.index, indices)
        self.assertEqual(lineset.nlines, len(indices))
        self.assertEqual(lineset.material, "mymaterial")

        # 序列化和反序列化
        lineset.save()
        loaded_lineset = collada.lineset.LineSet.load(collada, {'mylinevertsource': linefloatsrc},
                                                      fromstring(tostring(lineset.xmlnode)))

        # 检查反序列化版本是否具有相同的属性
        self.assertEqual(loaded_lineset.sources, lineset.sources)
        self.assertEqual(loaded_lineset.material, lineset.material)
        assert_array_equal(loaded_lineset.index, lineset.index)
        assert_array_equal(loaded_lineset.indices, lineset.indices)
        self.assertEqual(loaded_lineset.nindices, lineset.nindices)
        self.assertEqual(loaded_lineset.nlines, lineset.nlines)


if __name__ == '__main__':
    unittest.main()
