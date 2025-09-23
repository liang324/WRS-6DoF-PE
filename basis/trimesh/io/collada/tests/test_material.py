# 这段代码通过单元测试验证了 COLLADA 模块中与材质相关的基本功能,包括效果、CImage、表面、采样器等的创建、保存和加载.
# 通过这些测试,可以确保在对 COLLADA 文件进行读写操作时,相关数据能够正确地保存和恢复.


import os
import sys
import collada
from collada.util import unittest
from collada.xmlutil import etree
from collada.material import OPAQUE_MODE

fromstring = etree.fromstring
tostring = etree.tostring


class TestMaterial(unittest.TestCase):
    """
    测试Collada库中的材质模块

    :param self: 测试类的实例
    """
    def setUp(self):
        """
        初始化测试环境,包括创建虚拟的Collada对象和相关材质元素
        """
        self.dummy = collada.Collada(aux_file_loader=self.image_dummy_loader, validate_output=True)
        self.dummy_cimage = collada.material.CImage("yourcimage", "./whatever.tga", self.dummy)
        self.cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        self.dummy.images.append(self.dummy_cimage)
        self.dummy.images.append(self.cimage)
        self.othereffect = collada.material.Effect("othereffect", [], "phong")
        self.dummy.effects.append(self.othereffect)

    def test_effect_saving(self):
        """
        测试Effect对象的保存和加载

        - 创建一个Effect对象并验证其初始属性
        - 修改Effect对象的属性并保存
        - 加载保存后的Effect对象并验证其属性
        """
        effect = collada.material.Effect("myeffect", [], "phong",
                                         emission=(0.1, 0.2, 0.3, 1.0),
                                         ambient=(0.4, 0.5, 0.6, 1.0),
                                         diffuse=(0.7, 0.8, 0.9, 0.5),
                                         specular=(0.3, 0.2, 0.1, 1.0),
                                         shininess=0.4,
                                         reflective=(0.7, 0.6, 0.5, 1.0),
                                         reflectivity=0.8,
                                         transparent=(0.2, 0.4, 0.6, 1.0),
                                         transparency=0.9)

        self.assertEqual(effect.id, "myeffect")
        self.assertEqual(effect.shininess, 0.4)
        self.assertEqual(effect.reflectivity, 0.8)
        self.assertEqual(effect.transparency, 0.9)
        self.assertTupleEqual(effect.emission, (0.1, 0.2, 0.3, 1.0))
        self.assertTupleEqual(effect.ambient, (0.4, 0.5, 0.6, 1.0))
        self.assertTupleEqual(effect.diffuse, (0.7, 0.8, 0.9, 0.5))
        self.assertTupleEqual(effect.specular, (0.3, 0.2, 0.1, 1.0))
        self.assertTupleEqual(effect.reflective, (0.7, 0.6, 0.5, 1.0))
        self.assertTupleEqual(effect.transparent, (0.2, 0.4, 0.6, 1.0))
        self.assertEqual(effect.double_sided, False)
        self.assertEqual(effect.opaque_mode, OPAQUE_MODE.A_ONE)
        self.assertIsNotNone(str(effect))

        effect.id = "youreffect"
        effect.shininess = 7.0
        effect.reflectivity = 2.0
        effect.transparency = 3.0
        effect.emission = (1.1, 1.2, 1.3, 1.0)
        effect.ambient = (1.4, 1.5, 1.6, 1.0)
        effect.diffuse = (1.7, 1.8, 1.9, 1.0)
        effect.specular = (1.3, 1.2, 1.1, 1.0)
        effect.reflective = (1.7, 1.6, 1.5, 0.3)
        effect.transparent = (1.2, 1.4, 1.6, 1.0)
        effect.opaque_mode = OPAQUE_MODE.RGB_ZERO
        effect.double_sided = True
        effect.save()

        loaded_effect = collada.material.Effect.load(self.dummy, {}, fromstring(tostring(effect.xmlnode)))
        self.assertEqual(loaded_effect.id, "youreffect")
        self.assertEqual(loaded_effect.shininess, 7.0)
        self.assertEqual(loaded_effect.reflectivity, 2.0)
        self.assertEqual(loaded_effect.transparency, 3.0)
        self.assertTupleEqual(loaded_effect.emission, (1.1, 1.2, 1.3, 1.0))
        self.assertTupleEqual(loaded_effect.ambient, (1.4, 1.5, 1.6, 1.0))
        self.assertTupleEqual(loaded_effect.diffuse, (1.7, 1.8, 1.9, 1.0))
        self.assertTupleEqual(loaded_effect.specular, (1.3, 1.2, 1.1, 1.0))
        self.assertTupleEqual(loaded_effect.reflective, (1.7, 1.6, 1.5, 0.3))
        self.assertTupleEqual(loaded_effect.transparent, (1.2, 1.4, 1.6, 1.0))
        self.assertEqual(loaded_effect.opaque_mode, OPAQUE_MODE.RGB_ZERO)
        self.assertEqual(loaded_effect.double_sided, True)

    def image_dummy_loader(self, fname):
        """
        虚拟图像加载器,用于测试
        """
        return self.image_return

    def test_cimage_saving(self):
        """
        测试CImage对象的保存和加载

        - 创建一个CImage对象并验证其初始属性
        - 修改CImage对象的属性并保存
        - 加载保存后的CImage对象并验证其属性
        """
        self.image_return = None
        cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        self.assertEqual(cimage.id, "mycimage")
        self.assertEqual(cimage.path, "./whatever.tga")
        cimage.id = "yourcimage"
        cimage.path = "./next.tga"
        cimage.save()
        loaded_cimage = collada.material.CImage.load(self.dummy, {}, fromstring(tostring(cimage.xmlnode)))
        self.assertEqual(loaded_cimage.id, "yourcimage")
        self.assertEqual(loaded_cimage.path, "./next.tga")
        with self.assertRaises(collada.DaeBrokenRefError):
            loaded_cimage.data
        self.assertEqual(loaded_cimage.data, '')
        self.assertEqual(loaded_cimage.pilimage, None)
        self.assertEqual(loaded_cimage.uintarray, None)
        self.assertEqual(loaded_cimage.floatarray, None)
        self.assertIsNotNone(str(cimage))

    def test_cimage_data_loading(self):
        """
        测试CImage对象的数据加载

        - 验证测试数据文件的存在
        - 加载图像数据并验证其大小
        - 验证图像数据的格式和尺寸
        """
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        texture_file_path = os.path.join(data_dir, "duckCM.tga")
        self.assertTrue(os.path.isfile(texture_file_path), "Could not find data/duckCM.tga file for testing")

        texdata = open(texture_file_path, 'rb').read()
        self.assertEqual(len(texdata), 786476)

        self.image_return = texdata
        cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        image_data = cimage.data
        self.assertEqual(len(image_data), 786476)

        try:
            from PIL import Image as pil
        except ImportError:
            pil = None

        if pil is not None:
            pil_image = cimage.pilimage
            self.assertTupleEqual(pil_image.size, (512, 512))
            self.assertEqual(pil_image.format, "TGA")

            numpy_uints = cimage.uintarray
            self.assertTupleEqual(numpy_uints.shape, (512, 512, 3))

            numpy_floats = cimage.floatarray
            self.assertTupleEqual(numpy_uints.shape, (512, 512, 3))

    def test_surface_saving(self):
        """
        测试Surface对象的保存和加载

        - 创建一个Surface对象并验证其初始属性
        - 修改Surface对象的属性并保存
        - 加载保存后的Surface对象并验证其属性
        """
        cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        surface = collada.material.Surface("mysurface", cimage)
        self.assertEqual(surface.id, "mysurface")
        self.assertEqual(surface.image.id, "mycimage")
        self.assertEqual(surface.format, "A8R8G8B8")
        self.assertIsNotNone(str(surface))
        surface.id = "yoursurface"
        surface.image = self.dummy_cimage
        surface.format = "OtherFormat"
        surface.save()
        loaded_surface = collada.material.Surface.load(self.dummy, {}, fromstring(tostring(surface.xmlnode)))
        self.assertEqual(loaded_surface.id, "yoursurface")
        self.assertEqual(loaded_surface.image.id, "yourcimage")
        self.assertEqual(loaded_surface.format, "OtherFormat")

    def test_surface_empty(self):
        """
        测试Surface对象的异常情况

        - 测试加载不完整的Surface对象时抛出异常
        """
        surface1 = """
        <surface xmlns="http://www.collada.org/2005/11/COLLADASchema" type="2D">
        <init_from>file1-image</init_from>
        <format>A8R8G8B8</format>
        </surface>
        """
        self.assertRaises(collada.DaeIncompleteError, collada.material.Surface.load, self.dummy, {},
                          fromstring(surface1))

        surface2 = """
        <newparam xmlns="http://www.collada.org/2005/11/COLLADASchema" sid="file1-surface">
        <surface xmlns="http://www.collada.org/2005/11/COLLADASchema" type="2D">
        <init_from>file1-image</init_from>
        <format>A8R8G8B8</format>
        </surface>
        </newparam>
        """
        self.assertRaises(collada.DaeBrokenRefError, collada.material.Surface.load, self.dummy, {},
                          fromstring(surface2))

        surface3 = """
        <newparam xmlns="http://www.collada.org/2005/11/COLLADASchema" sid="file1-surface">
        <surface xmlns="http://www.collada.org/2005/11/COLLADASchema" type="2D">
        <init_from></init_from>
        <format>A8R8G8B8</format>
        </surface>
        </newparam>
        """
        self.assertRaises(collada.DaeBrokenRefError, collada.material.Surface.load, self.dummy, {},
                          fromstring(surface3))

    def test_sampler2d_saving(self):
        """
        测试 Sampler2D对象的保存和加载

        - 创建一个Sampler2D对象并验证其初始属性
        - 修改Sampler2D对象的属性并保存
        - 加载保存后的Sampler2D对象并验证其属性
        """
        cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        surface = collada.material.Surface("mysurface", cimage)
        sampler2d = collada.material.Sampler2D("mysampler2d", surface)
        self.assertEqual(sampler2d.id, "mysampler2d")
        self.assertEqual(sampler2d.minfilter, None)
        self.assertEqual(sampler2d.magfilter, None)
        self.assertEqual(sampler2d.surface.id, "mysurface")
        sampler2d = collada.material.Sampler2D("mysampler2d", surface, "LINEAR_MIPMAP_LINEAR", "LINEAR")
        self.assertEqual(sampler2d.minfilter, "LINEAR_MIPMAP_LINEAR")
        self.assertEqual(sampler2d.magfilter, "LINEAR")
        self.assertIsNotNone(str(sampler2d))

        other_surface = collada.material.Surface("yoursurface", cimage)
        sampler2d.id = "yoursampler2d"
        sampler2d.minfilter = "QUADRATIC_MIPMAP_WHAT"
        sampler2d.magfilter = "QUADRATIC"
        sampler2d.surface = other_surface
        sampler2d.save()

        loaded_sampler2d = collada.material.Sampler2D.load(self.dummy,
                                                           {'yoursurface': other_surface},
                                                           fromstring(tostring(sampler2d.xmlnode)))
        self.assertEqual(loaded_sampler2d.id, "yoursampler2d")
        self.assertEqual(loaded_sampler2d.surface.id, "yoursurface")
        self.assertEqual(loaded_sampler2d.minfilter, "QUADRATIC_MIPMAP_WHAT")
        self.assertEqual(loaded_sampler2d.magfilter, "QUADRATIC")

    def test_map_saving(self):
        """
        测试Map对象的保存和加载

        - 创建一个Map对象并验证其初始属性
        - 修改Map对象的属性并保存
        - 加载保存后的Map对象并验证其属性
        """
        cimage = collada.material.CImage("mycimage", "./whatever.tga", self.dummy)
        surface = collada.material.Surface("mysurface", cimage)
        sampler2d = collada.material.Sampler2D("mysampler2d", surface)
        map = collada.material.Map(sampler2d, "TEX0")
        self.assertEqual(map.sampler.id, "mysampler2d")
        self.assertEqual(map.texcoord, "TEX0")
        self.assertIsNotNone(str(map))

        other_sampler2d = collada.material.Sampler2D("yoursampler2d", surface)
        map.sampler = other_sampler2d
        map.texcoord = "TEX1"
        map.save()

        loaded_map = collada.material.Map.load(self.dummy,
                                               {'yoursampler2d': other_sampler2d}, fromstring(tostring(map.xmlnode)))
        self.assertEqual(map.sampler.id, "yoursampler2d")
        self.assertEqual(map.texcoord, "TEX1")

    def test_effect_with_params(self):
        """
        测试带有参数的Effect对象

        - 创建一个Effect对象并验证其初始属性
        - 修改Effect对象的参数并保存
        - 加载保存后的Effect对象并验证其参数
        """
        surface = collada.material.Surface("mysurface", self.cimage)
        sampler2d = collada.material.Sampler2D("mysampler2d", surface)
        effect = collada.material.Effect("myeffect", [surface, sampler2d], "phong",
                                         emission=(0.1, 0.2, 0.3, 1.0),
                                         ambient=(0.4, 0.5, 0.6, 1.0),
                                         diffuse=(0.7, 0.8, 0.9, 1.0),
                                         specular=(0.3, 0.2, 0.1, 1.0),
                                         shininess=0.4,
                                         reflective=(0.7, 0.6, 0.5, 1.0),
                                         reflectivity=0.8,
                                         transparent=(0.2, 0.4, 0.6, 1.0),
                                         transparency=0.9,
                                         opaque_mode=OPAQUE_MODE.A_ONE)

        other_cimage = collada.material.CImage("yourcimage", "./whatever.tga", self.dummy)
        other_surface = collada.material.Surface("yoursurface", other_cimage)
        other_sampler2d = collada.material.Sampler2D("yoursampler2d", other_surface)
        other_map = collada.material.Map(other_sampler2d, "TEX0")
        effect.params.pop()
        effect.params.append(other_surface)
        effect.params.append(other_sampler2d)
        effect.diffuse = other_map
        effect.transparent = other_map
        effect.save()

        self.dummy.images.append(self.dummy_cimage)
        loaded_effect = collada.material.Effect.load(self.dummy, {}, fromstring(tostring(effect.xmlnode)))
        self.assertEqual(type(loaded_effect.diffuse), collada.material.Map)
        self.assertEqual(type(loaded_effect.transparent), collada.material.Map)
        self.assertEqual(len(loaded_effect.params), 3)
        self.assertTrue(type(loaded_effect.params[0]) is collada.material.Surface)
        self.assertEqual(loaded_effect.params[0].id, "mysurface")
        self.assertTrue(type(loaded_effect.params[1]) is collada.material.Surface)
        self.assertEqual(loaded_effect.params[1].id, "yoursurface")
        self.assertTrue(type(loaded_effect.params[2]) is collada.material.Sampler2D)
        self.assertEqual(loaded_effect.params[2].id, "yoursampler2d")
        self.assertEqual(loaded_effect.opaque_mode, OPAQUE_MODE.A_ONE)

    def test_rgbzero(self):
        """
        测试Effect对象的RGB_ZERO模式

        - 创建一个Effect对象并验证其RGB_ZERO模式
        - 保存并加载Effect对象,验证其RGB_ZERO模式
        """
        effect = collada.material.Effect("myeffect", [], "phong",
                                         opaque_mode=OPAQUE_MODE.RGB_ZERO)

        self.assertEqual(effect.opaque_mode, OPAQUE_MODE.RGB_ZERO)
        self.assertEqual(effect.transparency, 0.0)
        effect.save()

        loaded_effect = collada.material.Effect.load(self.dummy, {}, fromstring(tostring(effect.xmlnode)))
        self.assertEqual(loaded_effect.opaque_mode, OPAQUE_MODE.RGB_ZERO)

        effect = collada.material.Effect("myeffect", [], "phong")

        self.assertEqual(effect.opaque_mode, OPAQUE_MODE.A_ONE)
        self.assertEqual(effect.transparency, 1.0)
        effect.save()

    def test_material_saving(self):
        """
        测试Material对象的保存和加载

        - 创建一个Material对象并验证其初始属性
        - 修改Material对象的属性并保存
        - 加载保存后的Material对象并验证其属性
        """
        effect = collada.material.Effect("myeffect", [], "phong")
        mat = collada.material.Material("mymaterial", "mymat", effect)
        self.assertEqual(mat.id, "mymaterial")
        self.assertEqual(mat.name, "mymat")
        self.assertEqual(mat.effect, effect)
        self.assertIsNotNone(str(mat))

        mat.id = "yourmaterial"
        mat.name = "yourmat"
        mat.effect = self.othereffect
        mat.save()

        loaded_mat = collada.material.Material.load(self.dummy, {}, fromstring(tostring(mat.xmlnode)))
        self.assertEqual(loaded_mat.id, "yourmaterial")
        self.assertEqual(loaded_mat.name, "yourmat")
        self.assertEqual(loaded_mat.effect.id, self.othereffect.id)


if __name__ == '__main__':
    unittest.main()
