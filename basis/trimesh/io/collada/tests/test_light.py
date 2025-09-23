# 这段代码通过单元测试验证了 COLLADA 模块中与光源相关的基本功能,包括方向光、环境光、点光和聚光灯的创建、保存和加载
# 通过这些测试,可以确保在对 COLLADA 文件进行读写操作时,光源的相关数据能够正确地保存和恢复

import collada
from collada.util import unittest
from collada.xmlutil import etree

fromstring = etree.fromstring
tostring = etree.tostring


class TestLight(unittest.TestCase):
    """
    测试Collada库中的光源模块

    :param self: 测试类的实例
    """
    def setUp(self):
        """
        初始化测试环境
        """
        self.dummy = collada.Collada(validate_output=True)

    def test_directional_light_saving(self):
        """
        测试方向光的保存和加载

        - 创建一个方向光对象并验证其属性
        - 修改方向光的颜色和ID,保存后重新加载并验证修改后的属性
        """
        dirlight = collada.light.DirectionalLight("mydirlight", (1, 1, 1))
        self.assertEqual(dirlight.id, "mydirlight")
        self.assertTupleEqual(dirlight.color, (1, 1, 1))
        self.assertTupleEqual(tuple(dirlight.direction), (0, 0, -1))
        self.assertIsNotNone(str(dirlight))
        dirlight.color = (0.1, 0.2, 0.3)
        dirlight.id = "yourdirlight"
        dirlight.save()
        loaded_dirlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(dirlight.xmlnode)))
        self.assertTrue(isinstance(loaded_dirlight, collada.light.DirectionalLight))
        self.assertTupleEqual(loaded_dirlight.color, (0.1, 0.2, 0.3))
        self.assertEqual(loaded_dirlight.id, "yourdirlight")

    def test_ambient_light_saving(self):
        """
        测试环境光的保存和加载

        - 创建一个环境光对象并验证其属性
        - 修改环境光的颜色和ID,保存后重新加载并验证修改后的属性
        """
        ambientlight = collada.light.AmbientLight("myambientlight", (1, 1, 1))
        self.assertEqual(ambientlight.id, "myambientlight")
        self.assertTupleEqual(ambientlight.color, (1, 1, 1))
        self.assertIsNotNone(str(ambientlight))
        ambientlight.color = (0.1, 0.2, 0.3)
        ambientlight.id = "yourambientlight"
        ambientlight.save()
        loaded_ambientlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(ambientlight.xmlnode)))
        self.assertTrue(isinstance(loaded_ambientlight, collada.light.AmbientLight))
        self.assertTupleEqual(ambientlight.color, (0.1, 0.2, 0.3))
        self.assertEqual(ambientlight.id, "yourambientlight")

    def test_point_light_saving(self):
        """
        测试点光源的保存和加载

        - 创建一个点光源对象并验证其属性
        - 修改点光源的颜色、衰减和ID,保存后重新加载并验证修改后的属性
        - 测试zfar属性的保存和加载
        """
        pointlight = collada.light.PointLight("mypointlight", (1, 1, 1))
        self.assertEqual(pointlight.id, "mypointlight")
        self.assertTupleEqual(pointlight.color, (1, 1, 1))
        self.assertEqual(pointlight.quad_att, None)
        self.assertEqual(pointlight.constant_att, None)
        self.assertEqual(pointlight.linear_att, None)
        self.assertEqual(pointlight.zfar, None)
        self.assertIsNotNone(str(pointlight))

        pointlight.color = (0.1, 0.2, 0.3)
        pointlight.constant_att = 0.7
        pointlight.linear_att = 0.8
        pointlight.quad_att = 0.9
        pointlight.id = "yourpointlight"
        pointlight.save()
        loaded_pointlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(pointlight.xmlnode)))
        self.assertTrue(isinstance(loaded_pointlight, collada.light.PointLight))
        self.assertTupleEqual(loaded_pointlight.color, (0.1, 0.2, 0.3))
        self.assertEqual(loaded_pointlight.constant_att, 0.7)
        self.assertEqual(loaded_pointlight.linear_att, 0.8)
        self.assertEqual(loaded_pointlight.quad_att, 0.9)
        self.assertEqual(loaded_pointlight.zfar, None)
        self.assertEqual(loaded_pointlight.id, "yourpointlight")

        loaded_pointlight.zfar = 0.2
        loaded_pointlight.save()
        loaded_pointlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(loaded_pointlight.xmlnode)))
        self.assertEqual(loaded_pointlight.zfar, 0.2)

    def test_spot_light_saving(self):
        """
        测试聚光灯的保存和加载

        - 创建一个聚光灯对象并验证其属性
        - 修改聚光灯的颜色、衰减、ID和其他属性,保存后重新加载并验证修改后的属性
        """
        spotlight = collada.light.SpotLight("myspotlight", (1, 1, 1))
        self.assertEqual(spotlight.id, "myspotlight")
        self.assertTupleEqual(spotlight.color, (1, 1, 1))
        self.assertEqual(spotlight.constant_att, None)
        self.assertEqual(spotlight.linear_att, None)
        self.assertEqual(spotlight.quad_att, None)
        self.assertEqual(spotlight.falloff_ang, None)
        self.assertEqual(spotlight.falloff_exp, None)
        self.assertIsNotNone(str(spotlight))

        spotlight.color = (0.1, 0.2, 0.3)
        spotlight.constant_att = 0.7
        spotlight.linear_att = 0.8
        spotlight.quad_att = 0.9
        spotlight.id = "yourspotlight"
        spotlight.save()
        loaded_spotlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(spotlight.xmlnode)))
        self.assertTrue(isinstance(loaded_spotlight, collada.light.SpotLight))
        self.assertTupleEqual(loaded_spotlight.color, (0.1, 0.2, 0.3))
        self.assertEqual(loaded_spotlight.constant_att, 0.7)
        self.assertEqual(loaded_spotlight.linear_att, 0.8)
        self.assertEqual(loaded_spotlight.quad_att, 0.9)
        self.assertEqual(loaded_spotlight.falloff_ang, None)
        self.assertEqual(loaded_spotlight.falloff_exp, None)
        self.assertEqual(loaded_spotlight.id, "yourspotlight")

        loaded_spotlight.falloff_ang = 180
        loaded_spotlight.falloff_exp = 2
        loaded_spotlight.save()
        loaded_spotlight = collada.light.Light.load(self.dummy, {}, fromstring(tostring(loaded_spotlight.xmlnode)))
        self.assertEqual(loaded_spotlight.falloff_ang, 180)
        self.assertEqual(loaded_spotlight.falloff_exp, 2)


if __name__ == '__main__':
    unittest.main()
