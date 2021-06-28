import argparse
import sympy
import numpy
import math
from typing import Optional, Union, List, Any
from prettytable import PrettyTable


def calculating_point_that_contains_PI(point: str) -> float:
    """ Функція для обчислення значення заданої точки, яка була отримана
        з командної строки, та яка має в собі математичну константу Пі в
        неявному виді, (наприк. Pi/4 чи pi * 2).

        Функція перетворює строкове значення заданної точки,
        в значення типу float.

    Args:
        point: задана точка з командної строки, яка має число Pi (тип string).
    Returns:


    """
    parts_point: List[Union[str, float]]
    value_point: float

    # Розбиваємо рядок зі введеним значенням заданої точки на список
    # двух елементів. Замінюємо строковий елемент "pi" на числовий елемент pi
    # типу float бібліотеки math. Обчислуємо і повертаємо значення точки.
    if '/' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('/')]
        try:
            value_point = float(parts_point[0]) / float(parts_point[1])
        except ValueError:
            error_messege = 'Помилка: точка повинна містити числа, не' + \
                            'враховуючи числа Pi.'
            raise argparse.ArgumentTypeError(error_messege)
        except ZeroDivisionError:
            error_messege = 'Помилка: {} - ділення на нуль!'.format(point)
            raise argparse.ArgumentTypeError(error_messege)
    elif '*' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('*')]
        try:
            value_point = float(parts_point[0]) * float(parts_point[1])
        except ValueError:
            error_messege = 'Помилка: точка повинна містити числа, не' + \
                            'враховуючи числа Pi.'
            raise argparse.ArgumentTypeError(error_messege)
    elif '+' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('+')]
        try:
            value_point = float(parts_point[0]) + float(parts_point[1])
        except ValueError:
            error_messege = 'Помилка: точка повинна містити числа, не' + \
                            'враховуючи числа Pi.'
            raise argparse.ArgumentTypeError(error_messege)
    elif '-' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('-')]
        try:
            value_point = float(parts_point[0]) - float(parts_point[1])
        except ValueError:
            error_messege = 'Помилка: точка повинна містити числа, не' + \
                            'враховуючи числа Pi.'
            raise argparse.ArgumentTypeError(error_messege)

    return value_point


def point_validation(point):
    """ A
    """

    point = point.replace(' ', '')

    if 'pi' in point:
        return calculating_point_that_contains_PI(point)
    else:
        if '/' in point:
            point = point.split('/')
            try:
                point = float(point[0]) / float(point[1])
                return point
            except ValueError or ZeroDivisionError:
                raise argparse.ArgumentTypeError
        else:
            try:
                point = float(point)
                return point
            except ValueError:
                raise argparse.ArgumentTypeError


parser = argparse.ArgumentParser(description='')
parser.add_argument('--step', '-h_val', type=float, default=1,
                    help='крок сітки, тип - float.')
parser.add_argument('--point', '-x0', type=point_validation, default=math.pi/4,
                    help='початкова точка. тип - float.')
parser.add_argument('--r_value', '-r', type=float, default=0.5,
                    help='початкова точка. тип - float.')
args = parser.parse_args()


def funciton(x: float) -> float:
    return math.cos(2 * x)


def find_derivative(x0: Optional[float] = None) -> Union[sympy.core.mul.Mul,
                                                         numpy.float64]:
    x = sympy.symbols('x')
    dx = sympy.cos(2 * x).diff(x)

    if x0 is None:
        return dx
    else:
        dx0 = sympy.lambdify(x, dx)
        return dx0(x0)


def print_table(name1: str, name2: str,
                nodes: List[float], y_val: List[float]) -> None:

    table = PrettyTable()
    table.add_column(name1, nodes)
    table.add_column(name2, y_val)

    print(table)


def main():

    # Вхідні дані
    h: int
    x0: int
    r: int

    h = args.step
    x0 = args.point
    r = args.r_value

    # Вузлові значення аргументу і функції (h крок)
    print('\n a) Вузлові значення аргументу і функції, ',
          'що диференціюється (h крок):\n')
    nodes = [round(x, 3) for x in numpy.arange(x0 - h, x0 + h + h, h)]
    y_val = [round(funciton(x), 3) for x in nodes]

    # Відображення таблиці
    print_table('Nodes (h)', 'Y-values (h)', nodes, y_val)

    # Обчислення значення похідної y(h)
    y0h = 1 / (2 * h) * (y_val[2] - y_val[0])
    print('\n\n б) Значення похідної:\n\n    ',
          'y\u2080\u2032(h) = {}\n'.format(y0h))

    # Вузлові значення аргументу і функції (rh крок)
    nodes_r = [round(x, 3) for x in
               numpy.arange(x0 - h * r, x0 + 2 * (h * r), h * r)]
    y_val_r = [round(funciton(x), 3) for x in nodes_r]

    # Відображення таблиці
    print('\n в) Вузлові значення аргументу і функції, ',
          'що диференціюється (rh крок):\n')
    print_table('Nodes (rh)', 'Y-value (rh)', nodes_r, y_val_r)

    # Обчислення значення похідної y(rh)
    y0hr = 1 / (2 * r * h) * (y_val_r[2] - y_val_r[0])
    print('\n\n г) Значення похідної:\n\n    ',
          'y\u2080\u2032(rh) = {}\n'.format(round(y0hr, 3)))

    # Обчислюється похибка значення похідної
    # за допомогою другої формули Рунге-Ромберга
    print('\n д) Похибка за другою формулою Рунге-Ромберга:\n')

    Oh2 = (y0h - y0hr) / (r**2 - 1)
    print('    O(h\u00B2) = {}'.format(round(Oh2, 3)))

    Oh2r = r**2 * Oh2 + y0hr
    print('    O(rh\u00B2) = {}'.format(round(Oh2r, 3)))

    # Результат 4-го порядку точності
    print('\n\n е) Значення похідної з підвищеним порядком точності ',
          '(4-им порядком):\n')

    Oh3 = (r**2 * y0h - y0hr) / (r**2 - 1)
    print('    y\u2080\u2032(O(h\u00B3)) = {}'.format(round(Oh3, 3)))

    # Точне значення похідної
    print('\n\n ж) Точне значення похідної:\n')
    print('    y\u2032(x) = {}'.format(derivative()))

    yx0 = derivative(x0)
    print('    y\u2032(x\u2080) = {}\n'.format(round(yx0, 3)))

    delta1 = y0h - yx0
    print('    \u03B4\u2081 = {}'.format(round(delta1, 3)))

    delta2 = y0hr - yx0
    print('    \u03B4\u2082 = {}'.format(round(delta2, 3)))

    delta3 = Oh3 - yx0
    print('    \u03B4\u2083 = {}\n\n'.format(round(delta3, 3)))




if __name__ == '__main__':

    print(calculating_point_that_contains_PI('pi/w'))
   # print(point_validation('pi / 0'))
   # main()

