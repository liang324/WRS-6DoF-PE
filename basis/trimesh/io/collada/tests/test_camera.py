#  这段代码是一个 Python 单元测试,用于测试 basis.trimesh.io.collada 模块中与 COLLADA 文件 camera 相关的类
#  特别是 PerspectiveCamera (透视相机) 和 OrthographicCamera (正交相机) 类的参数组合和保存/加载功能

import collada
from collada.common import DaeMalformedError
from collada.util import unittest
from collada.xmlutil import etree

fromstring = etree.fromstring
tostring = etree.tostring


class TestCamera(unittest.TestCase):
    """
    测试 `collada.camera` 模块中的相机功能

    :param dummy: 一个 `Collada` 对象,用于测试环境的初始化
    """
    def setUp(self):
        """
        设置测试环境,初始化 COLLADA 对象
        """
        self.dummy = collada.Collada(validate_output=True)

    def test_perspective_camera_xfov_yfov_aspect_ratio(self):
        """
        测试透视相机的 `xfov`、`yfov` 和 `aspect_ratio` 属性组合的有效性

        :raises DaeMalformedError: 当属性组合无效时抛出
        """
        # 测试无效的 `xfov`、`yfov`、`aspect_ratio` 组合
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=None, yfov=None, aspect_ratio=None)
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=0.2, yfov=30, aspect_ratio=50)
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=None, yfov=None, aspect_ratio=50)

        # xfov alone
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=30, yfov=None, aspect_ratio=None)
        self.assertEqual(cam.xfov, 30)
        self.assertIsNone(cam.yfov)
        self.assertIsNone(cam.aspect_ratio)

        # yfov alone
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=None, yfov=50, aspect_ratio=None)
        self.assertIsNone(cam.xfov)
        self.assertEqual(cam.yfov, 50)
        self.assertIsNone(cam.aspect_ratio)

        # xfov + yfov
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=30, yfov=50, aspect_ratio=None)
        self.assertEqual(cam.xfov, 30)
        self.assertEqual(cam.yfov, 50)
        self.assertIsNone(cam.aspect_ratio)

        # xfov + aspect_ratio
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=30, yfov=None, aspect_ratio=1)
        self.assertEqual(cam.xfov, 30)
        self.assertIsNone(cam.yfov)
        self.assertEqual(cam.aspect_ratio, 1)

        # yfov + aspect_ratio
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=None, yfov=50, aspect_ratio=1)
        self.assertIsNone(cam.xfov)
        self.assertEqual(cam.yfov, 50)
        self.assertEqual(cam.aspect_ratio, 1)

    def test_perspective_camera_saving(self):
        """
        测试透视相机的保存和加载功能

        :return: 验证相机属性在保存和加载后保持一致
        """
        cam = collada.camera.PerspectiveCamera("mycam", 1, 1000, xfov=30)
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xfov, 30)
        self.assertEqual(cam.yfov, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam.save()
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xfov, 30)
        self.assertEqual(cam.yfov, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam = collada.camera.PerspectiveCamera.load(self.dummy, {}, fromstring(tostring(cam.xmlnode)))
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xfov, 30)
        self.assertEqual(cam.yfov, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam.id = "yourcam"
        cam.znear = 5
        cam.zfar = 500
        cam.xfov = None
        cam.yfov = 50
        cam.aspect_ratio = 1.3
        cam.save()
        cam = collada.camera.PerspectiveCamera.load(self.dummy, {}, fromstring(tostring(cam.xmlnode)))
        self.assertEqual(cam.id, "yourcam")
        self.assertEqual(cam.znear, 5)
        self.assertEqual(cam.zfar, 500)
        self.assertEqual(cam.xfov, None)
        self.assertEqual(cam.yfov, 50)
        self.assertEqual(cam.aspect_ratio, 1.3)

        cam.xfov = 20
        with self.assertRaises(DaeMalformedError):
            cam.save()

    def test_orthographic_camera_xmag_ymag_aspect_ratio(self):
        """
        测试正交相机的 `xmag`、`ymag` 和 `aspect_ratio` 属性组合的有效性

        :raises DaeMalformedError: 当属性组合无效时抛出
        """
        # 测试无效的 `xmag`、`ymag`、`aspect_ratio` 组合
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=None, ymag=None, aspect_ratio=None)
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=0.2, ymag=30, aspect_ratio=50)
        with self.assertRaises(DaeMalformedError):
            cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=None, ymag=None, aspect_ratio=50)

        # xmag alone
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=30, ymag=None, aspect_ratio=None)
        self.assertEqual(cam.xmag, 30)
        self.assertIsNone(cam.ymag)
        self.assertIsNone(cam.aspect_ratio)

        # ymag alone
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=None, ymag=50, aspect_ratio=None)
        self.assertIsNone(cam.xmag)
        self.assertEqual(cam.ymag, 50)
        self.assertIsNone(cam.aspect_ratio)

        # xmag + ymag
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=30, ymag=50, aspect_ratio=None)
        self.assertEqual(cam.xmag, 30)
        self.assertEqual(cam.ymag, 50)
        self.assertIsNone(cam.aspect_ratio)

        # xmag + aspect_ratio
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=30, ymag=None, aspect_ratio=1)
        self.assertEqual(cam.xmag, 30)
        self.assertIsNone(cam.ymag)
        self.assertEqual(cam.aspect_ratio, 1)

        # ymag + aspect_ratio
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=None, ymag=50, aspect_ratio=1)
        self.assertIsNone(cam.xmag)
        self.assertEqual(cam.ymag, 50)
        self.assertEqual(cam.aspect_ratio, 1)

    def test_orthographic_camera_saving(self):
        """
        测试正交相机的保存和加载功能

        :return: 验证相机属性在保存和加载后保持一致
        """
        cam = collada.camera.OrthographicCamera("mycam", 1, 1000, xmag=30)
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xmag, 30)
        self.assertEqual(cam.ymag, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam.save()
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xmag, 30)
        self.assertEqual(cam.ymag, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam = collada.camera.OrthographicCamera.load(self.dummy, {}, fromstring(tostring(cam.xmlnode)))
        self.assertEqual(cam.id, "mycam")
        self.assertEqual(cam.znear, 1)
        self.assertEqual(cam.zfar, 1000)
        self.assertEqual(cam.xmag, 30)
        self.assertEqual(cam.ymag, None)
        self.assertEqual(cam.aspect_ratio, None)

        cam.id = "yourcam"
        cam.znear = 5
        cam.zfar = 500
        cam.xmag = None
        cam.ymag = 50
        cam.aspect_ratio = 1.3
        cam.save()
        cam = collada.camera.OrthographicCamera.load(self.dummy, {}, fromstring(tostring(cam.xmlnode)))
        self.assertEqual(cam.id, "yourcam")
        self.assertEqual(cam.znear, 5)
        self.assertEqual(cam.zfar, 500)
        self.assertEqual(cam.xmag, None)
        self.assertEqual(cam.ymag, 50)
        self.assertEqual(cam.aspect_ratio, 1.3)

        cam.xmag = 20
        with self.assertRaises(DaeMalformedError):
            cam.save()


if __name__ == '__main__':
    unittest.main()
