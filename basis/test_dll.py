# 这段代码是一个将 C++ 编写的高性能旋转矩阵函数集成进 Python 并进行性能对比的完整示例

import ctypes
import numpy as np
import basis.robot_math as rm

# CPP
# https://blog.janjan.net/2021/01/26/vsc-python-import-windows-dll-error-function-not-found/

# 查找共享库,路径取决于平台和 Python 版本
# 1. 打开共享库
robot_math_c = ctypes.cdll.LoadLibrary("./robotmath_fast_ctype.dll")

# 2. 告诉 Python 函数的参数和返回类型
robot_math_c.rotmat_from_axangle.restype = np.ctypeslib.ndpointer(dtype=np.double, shape=(3, 3), flags="C_CONTIGUOUS")
robot_math_c.rotmat_from_axangle.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, shape=(1, 3), flags="C_CONTIGUOUS"),
    ctypes.c_double]

# 3. 调用函数并进行性能测试
import timeit

# 使用 C 函数计算旋转矩阵,并统计 12000 次调用所需时间
t = timeit.timeit(lambda: robot_math_c.rotmat_from_axangle(np.array([[0, 1, 0]], dtype=np.double), np.pi / 3),
                  number=12000)
print(t)

# 使用 Python 实现的函数进行相同操作,比较性能
t = timeit.timeit(lambda: rm.rotmat_from_axangle(np.array([0, 1, 0], dtype=np.double), np.pi / 3), number=12000)
print(t)
