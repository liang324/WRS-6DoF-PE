import numpy as np
import vision.depth_camera.surface._surface as sfc
from scipy.optimize import curve_fit


class MixedGaussianSurface(sfc.Surface):
    def __init__(self,
                 xydata,
                 zdata,
                 n_mix=1,
                 init_guess=[0, 0, .05, .05, .01]):
        """
        表示一个混合高斯曲面的类,继承自 Surface 类

        :param xydata: 二维数组,包含表面上的 (x, y) 坐标
        :param zdata: 一维数组,包含对应的 z 坐标数据
        :param n_mix: 混合的高斯函数的数量
        :param init_guess: 初始猜测参数列表,每个高斯函数包含 [x_mean, y_mean, x_delta, y_delta, attitude]

        author: weiwei
        date: 20210624
        """
        super().__init__(xydata, zdata)
        # 使用 curve_fit 函数拟合混合高斯模型
        guess_prms = np.array([init_guess] * n_mix)
        self.popt, pcov = curve_fit(MixedGaussianSurface.mixed_gaussian, xydata, zdata, guess_prms.ravel())

    @staticmethod
    def mixed_gaussian(xydata, *parameters):
        """
        混合高斯函数,用于计算给定 (x, y) 坐标的 z 值

        :param xydata: 二维数组,包含要计算的 (x, y) 坐标
        :param parameters: 高斯函数的参数,每个高斯函数包含 [x_mean, y_mean, x_delta, y_delta, attitude]
        :return: 一维数组,包含计算得到的 z 坐标

        author: weiwei
        date; 20210624
        """
        def gaussian(xdata, ydata, xmean, ymean, xdelta, ydelta, attitude):
            return attitude * np.exp(-((xdata - xmean) / xdelta) ** 2 - ((ydata - ymean) / ydelta) ** 2)

        z = np.zeros(len(xydata))
        for single_parameters in np.array(parameters).reshape(-1, 5):
            z += gaussian(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        """
        根据给定的 (x, y) 坐标数据计算 z 坐标

        :param xydata: 二维数组,包含要计算的 (x, y) 坐标
        :return: 一维数组,包含计算得到的 z 坐标
        """
        zdata = MixedGaussianSurface.mixed_gaussian(xydata, self.popt)
        return zdata


if __name__ == '__main__':
    import numpy as np
    # from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    # 定义拟合的二维域
    xmin, xmax, nx = -5, 4, 75
    ymin, ymax, ny = -3, 7, 150
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)


    # 我们要拟合的函数是二维高斯函数的和
    def gaussian(x, y, x0, y0, xalpha, yalpha, A):
        return A * np.exp(-((x - x0) / xalpha) ** 2 - ((y - y0) / yalpha) ** 2)


    # 高斯参数列表:x0, y0, xalpha, yalpha, A
    gprms = [(0, 2, 2.5, 5.4, 1.5),
             (-1, 4, 6, 2.5, 1.8),
             (-3, -0.5, 1, 2, 4),
             (3, 0.5, 2, 1, 5)]
    # 添加噪声的标准差,用于生成我们的测试函数进行拟合
    noise_sigma = 0.1
    # 要拟合的函数是 Z
    Z = np.zeros(X.shape)
    for p in gprms:
        Z += gaussian(X, Y, *p)
    Z += noise_sigma * np.random.randn(*Z.shape)

    # 绘制拟合函数和残差的3D图形
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')  # 使用 add_subplot 而不是 gca
    # ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_zlim(0, np.max(Z) + 2)
    plt.show()

    # 将 X 和 Y 数据组合成 xdata
    xdata = np.vstack((X.ravel(), Y.ravel())).T
    import visualization.panda.world as wd

    # 创建 Panda3D 世界并设置相机位置
    base = wd.World(cam_pos=np.array([7, 7, 20]), lookat_pos=np.array([0, 0, 0.05]))

    # 创建混合高斯曲面并生成几何模型
    surface = MixedGaussianSurface(xdata, Z.ravel())

    # 将几何模型附加到 Panda3D 世界
    surface_gm = surface.get_gometricmodel()
    surface_gm.attach_to(base)
    base.run()
