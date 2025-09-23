import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as sympy_parse
from .constants import log
from . import util


def symbolic_barycentric(function):
    '''
    符号化地对函数(x, y, z)在三角形或网格上进行积分

    :param function: 字符串或 sympy 表达式
                     x, y, z 将被替换为重心表示
                     并且函数将在三角形上积分
    :return:
        evaluator: numpy lambda 函数,结果用于计算网格
        expr: sympy 表达式,结果
    '''

    class evaluator:
        def __init__(self, expr, expr_args):
            self.lambdified = sp.lambdify(args=expr_args,
                                          expr=expr,
                                          modules='numpy')

        def __call__(self, mesh):
            '''
            快速评估网格上的表面积分

            :param mesh: Trimesh 对象
            :return: integrated: (len(faces),) float,每个面的积分结果
            '''
            integrated = self.lambdified(*mesh.triangles.reshape((-1, 9)).T)
            integrated *= 2 * mesh.area_faces
            return integrated

    if util.is_string(function):
        function = sympy_parse(function)
    # 重心坐标
    b1, b2 = sp.symbols('b1 b2', real=True, positive=True)
    # 三角形的顶点
    x1, x2, x3, y1, y2, y3, z1, z2, z3 = sp.symbols('x1,x2,x3,y1,y2,y3,z1,z2,z3', real=True)

    # 生成替换字典,将笛卡尔坐标转换为重心坐标
    # 因为输入可以是 sympy 表达式或我们解析的字符串
    # 基于名称替换以避免 id(x) 问题
    substitutions = {}
    for symbol in function.free_symbols:
        if symbol.name == 'x':
            substitutions[symbol] = b1 * x1 + b2 * x2 + (1 - b1 - b2) * x3
        elif symbol.name == 'y':
            substitutions[symbol] = b1 * y1 + b2 * y2 + (1 - b1 - b2) * y3
        elif symbol.name == 'z':
            substitutions[symbol] = b1 * z1 + b2 * z2 + (1 - b1 - b2) * z3

    # 应用转换到重心坐标
    function = function.subs(substitutions)
    log.debug('转换函数为重心坐标: %s', str(function))

    # 对 b1 进行第一次积分
    integrated_1 = sp.integrate(function, b1)
    integrated_1 = (integrated_1.subs({b1: 1 - b2}) - integrated_1.subs({b1: 0}))
    integrated_2 = sp.integrate(integrated_1, b2)
    integrated_2 = (integrated_2.subs({b2: 1}) - integrated_2.subs({b2: 0}))
    lambdified = evaluator(expr=integrated_2, expr_args=[x1, y1, z1, x2, y2, z2, x3, y3, z3])
    return lambdified, integrated_2
