import argparse
import re
import sympy as sym
import numpy
import math
from typing import Optional, Tuple, Union, List, Dict, Callable
from prettytable import PrettyTable


def calculating_point_that_contains_PI(point: str) -> float:
    """ Функція для обчислення значення заданої точки, яка була отримана
        з командної строки, та яка містить математичну константу Pі у
        неявному вигляді.

        Функція перетворює строкове значення заданої точки в значення
        типу float.

        Приклади роботи функції:

        'pi/4' -> 0.78539816339
        '2/pi' -> 0.63661977236
        'pi*3' -> 9.42477796077
        '4+pi' -> 7.14159265359
        'pi-1' -> 2.14159265359

    Args:
        point: задана точка з командної строки, яка містить Pi (тип - string).
    Returns:
        обчислене значення заданої точки.
    """

    parts_point: List[Union[str, float]]
    value_point: float

    # Розбиваємо рядок з введеним значенням заданої точки на список
    # двух елементів. Замінюємо строковий елемент "pi" на числовий елемент pi
    # типу float бібліотеки math. Обчислуємо і повертаємо значення точки.
    if '/' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('/')]
        try:
            value_point = float(parts_point[0]) / float(parts_point[1])
        except ValueError:
            raise argparse.ArgumentTypeError
        except ZeroDivisionError:
            raise argparse.ArgumentTypeError
    elif '*' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('*')]
        try:
            value_point = float(parts_point[0]) * float(parts_point[1])
        except ValueError:
            raise argparse.ArgumentTypeError
    elif '+' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('+')]
        try:
            value_point = float(parts_point[0]) + float(parts_point[1])
        except ValueError:
            raise argparse.ArgumentTypeError
    elif '-' in point:
        parts_point = [math.pi if x == 'pi' else x for x in point.split('-')]
        try:
            value_point = float(parts_point[0]) - float(parts_point[1])
        except ValueError:
            raise argparse.ArgumentTypeError

    return value_point


def point_validation(point: str) -> Tuple[float, str]:
    """ Функція для валідації заданого значення точки з командної строки.

    Якщо значення містить в собі константу Pi, вичисляємо його в
    функції `calculating_point_that_contains_PI` та повертаємо значення.
    Якщо ні, робимо перевірку вхідного значення та повертаємо його.

    Parameters
    ----------
        point: `str`
            Задана точка з командної строки.

    Returns
    -------
    class: `Tuple[float, str]`

        Повертається кортеж з двома елементами:
        - `value_point` - обчислене значення точки.
        - `original_point` - не змінене значення точки.

    Raises
    ------
    ValueError
        Якщо значення точки - `point`, яке не містить в собі константу Pi
        неможливо перетворити в число з плаваючою точкою (`float`).
    """

    value_point: float
    point = point.replace(' ', '')

    original_point: str = point

    if 'pi' in point:
        value_point = calculating_point_that_contains_PI(point)
        return (value_point, original_point)

    try:
        value_point = float(point)
        return (value_point, original_point)
    except ValueError:
        raise argparse.ArgumentTypeError


parser = argparse.ArgumentParser(description='Обчислення похідної.')
parser.add_argument('--function', '-f', type=str, default='ctg(x)',
                    choices=['sin(x)', 'sin(2x)', 'sin(3x)', 'sin(0.5x)',
                             'cos(x)', 'cos(2x)', 'cos(3x)', 'cos(0.5x)',
                             'tg(x)', 'tg(2x)', 'tg(3x)', 'tg(0.5x)',
                             'ctg(x)', 'ctg(2x)', 'ctg(3x)', 'ctg(0.5x)',
                             'ln(x)', 'ln(x^2)', 'ln(x+2)', '1/x', '1/(2x)',
                             '2/x', '1/(x^2)', '1/(x^3)', '1/(x+1)',
                             '1/(x^2+1)', '1/(x^3+1)', 'SQRT(x)', 'SQRT(2x)',
                             'SQRT(0.5x)', 'SQRT(x+1)', 'SQRT(2x+1)',
                             'SQRT(0.5x+1)', '1/SQRT(x)'],
                    help='Функція для диференціювання. Приклад: \'ln(x^2)\'')
parser.add_argument('--step', '-h_val', type=float, default=1,
                    help='Крок сітки. Тип - float.')
parser.add_argument('--point', '-x0', type=point_validation, default=math.pi/4,
                    help='Початкова точка. Приклад: pi/4, 2*pi, 3, 1-pi, 8.')
parser.add_argument('--r_value', '-r', type=float, default=0.5,
                    help='Початкова точка. Тип - float.')
args = parser.parse_args()


def get_function_value(x: float,
                       function: str = args.function) -> Optional[float]:
    """Функція для обчислення значення заданої функції у точках х.

    Функія працює лише з деякими математичними функціями, які представлені
    у методичці.

    Parameters
    ----------
        - x: `float`

        Значення точки.

        - function: `str`

        Задана функція з командної строки.

    Returns
    -------
    - class: `float`

        Значення заданої функції у точці х.
    """

    available_functions: Dict[str, Callable]
    available_functions = {
            'sin(x)': lambda x: math.sin(x),
            'sin(2x)': lambda x: math.sin(2 * x),
            'sin(3x)': lambda x: math.sin(3*x),
            'sin(0.5x)': lambda x: math.sin(0.5 * x),
            'cos(x)': lambda x: math.cos(x),
            'cos(2x)': lambda x: math.cos(2 * x),
            'cos(3*x)': lambda x: math.cos(3 * x),
            'cos(0.5x)': lambda x: math.cos(0.5 * x),
            'tg(x)': lambda x: math.tan(x),
            'tg(2x)': lambda x: math.tan(2 * x),
            'tg(3x)': lambda x: math.tan(3 * x),
            'tg(0.5x)': lambda x: math.tan(0.5 * x),
            'ctg(x)': lambda x: math.cos(x) / math.sin(x),
            'ctg(2x)': lambda x: math.cos(2 * x) / math.sin(2 * x),
            'ctg(3x)': lambda x: math.cos(3 * x) / math.sin(3 * x),
            'ctg(0.5x)': lambda x: math.cos(0.5 * x) / math.sin(0.5 * x),
            'ln(x)': lambda x: math.log10(x),
            'ln(x^2)': lambda x: math.log10(x**2),
            'ln(x+2)': lambda x: math.log10(x + 2),
            '1/x': lambda x: 1 / x,
            '1/(2x)': lambda x: 1 / (2 * x),
            '2/x': lambda x: 2 / x,
            '1/(x^2)': lambda x: 1 / x**2,
            '1/(x^3)': lambda x: 1 / x**3,
            '1/(x+1)': lambda x: 1 / (x + 1),
            '1/(x^2+1)': lambda x: 1 / (x**2 + 1),
            '1/(x^3+1)': lambda x: 1 / (x**3 + 1),
            'SQRT(x)': lambda x: math.sqrt(x),
            'SQRT(2x)': lambda x: math.sqrt(2 * x),
            'SQRT(0.5x)': lambda x: math.sqrt(0.5 * x),
            'SQRT(x+1)': lambda x: math.sqrt(x + 1),
            'SQRT(2x+1)': lambda x: math.sqrt(2 * x + 1),
            'SQRT(0.5x+1)': lambda x: math.sqrt(0.5 * x + 1),
            '1/SQRT(x)': lambda x: 1 / math.sqrt(x)
    }

    return float(available_functions[function](x))


def transformation_string_function(function: str) -> Optional[str]:
    """Функція для перетворення рядка, який мість задану функцію.

    Якщо функція, яка міститься у рядку - неявно задана для модуля `sympy`,
    ми перетворюємо її у функцію, яка буде валідна для модуля.

    Parameters
    ----------
    - function: `str`

    Рядок, який містить задану математичну функцію.

    Examples
    --------
    +------------------+------------------------+
    | Рядок з функцією | Результат перетворення |
    +==================+========================+
    |      sin(2x)     |        sin(2*x)        |
    +------------------+------------------------+
    |      cos(3x)     |        cos(3*x)        |
    +------------------+------------------------+
    |      tg(0.5x)    |        tan(0.5*x)      |
    +------------------+------------------------+
    |      ctg(x)      |        cot(x)          |
    +------------------+------------------------+
    |      ctg(3x)     |        cot(3*x)        |
    +------------------+------------------------+
    |      ln(x^2)     |        ln(x**2)        |
    +------------------+------------------------+
    |      1/(x^3+1)   |        1/(x**3+1)      |
    +------------------+------------------------+
    |      SQRT(2x)    |        sqrt(2*x)       |
    +------------------+------------------------+

    Returns
    -------
    - class: `str`, optional

    Рядок зі зміненою функцією.
    """

    if re.findall(r'\d[x]', function):
        function = '*'.join(re.split(r'(?=[x])', function))
        return function

    if function.startswith('ctg'):
        return function.replace('ctg', 'cot')
    elif function.startswith('ln'):
        return function.replace('^', '**')
    elif function.startswith('tg'):
        return function.replace('tg', 'tan')
    elif function.startswith('SQRT'):
        return function.replace('SQRT', 'sqrt')

    return None


def find_derivative(function: Optional[str] =
                    transformation_string_function(args.function),
                    x0: Optional[float] = None) -> Union[str, float]:
    """Функція для знаходження похідної заданої функції.

    Якщо початкова точка - `None`, повертається похідна функції.
    Якщо початкова точка передана в функцію, обчислюється
    похнідна функції у початковій точці.

    Parameters
    ----------
        - function : `str`,  optional

        Задана функція з командної строки.

        - x0 : `float`, optional

        Задана початкова точка з командної строки.

    Returns
    -------
    -class: Union[`str`, `float`]

    Повертає обчислену похідну функції, якщо `x0` - `None`.
    Інакше, повертає обчилсену похідну функції у точці `x0`.
    """

    x = sym.symbols('x')

    if x0 is None:
        return str(sym.diff(function))
    else:
        return float(sym.diff(function).evalf(subs={x: x0}))


def print_table(name1: str, name2: str,
                nodes: List[float], y_val: List[float]) -> None:
    """ Функція для створення та відображення таблиці за допомогою
        модуля `prettytable`.

        Таблиця використовується для вузлових значень аргументу і функції.

    Parameters
    ----------
        - name1 : `str`

        Назва першого стовпчика таблиці.

        - name2 : `str`

        Назва другого стовпчика таблиці.

        - nodes : List[`float`]

        Значення вузлів.

        - y_val : List[`float`]

        Значення функції.
    """

    table = PrettyTable()
    table.add_column(name1, nodes)
    table.add_column(name2, y_val)

    print(table)

    return None


def show_input_data(func: str, step: float, x0: float, r_value: float) -> None:
    """ Функція для відображення вхідних данних.

    """

    print('\n Вхідні дані:\n\n',
          f' Функція: {func}\n',
          f' Крок \'h\' = {step}\n',
          f' Початкова точка = {x0}\n',
          f' Крок \'r\' = {r_value}\n')

    return None


def main():

    # Вхідні дані
    func: str = args.function
    h: float = args.step
    x0: float = args.point[0]
    original_x0: str = args.point[1]
    r: float = args.r_value

    show_input_data(func, h, original_x0, r)

    # Вузлові значення аргументу і функції (h крок)
    print('\n a) Вузлові значення аргументу і функції, ',
          'що диференціюється (h крок):\n')
    nodes = [round(x, 3) for x in numpy.arange(x0 - h, x0 + h + h, h)]
    y_val = [round(get_function_value(x), 3) for x in nodes]

    # Відображення таблиці
    print_table('Nodes (h)', 'Y-values (h)', nodes, y_val)

    # Обчислення значення похідної y(h)
    y0h = 1 / (2 * h) * (y_val[2] - y_val[0])
    print('\n\n б) Значення похідної:\n\n    ',
          'y\u2080\u2032(h) = {}\n'.format(y0h))

    # Вузлові значення аргументу і функції (rh крок)
    nodes_r = [round(x, 3) for x in
               numpy.arange(x0 - h * r, x0 + 2 * (h * r), h * r)]
    y_val_r = [round(get_function_value(x), 3) for x in nodes_r]

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

    # Точне значення похідної`
    print('\n\n ж) Точне значення похідної:\n')
    print('    y\u2032(x) = {}'.format(find_derivative()))

    yx0 = find_derivative(x0=x0)
    print('    y\u2032(x\u2080) = {}\n'.format(round(yx0, 3)))

    delta1 = y0h - yx0
    print('    \u03B4\u2081 = {}'.format(round(delta1, 3)))

    delta2 = y0hr - yx0
    print('    \u03B4\u2082 = {}'.format(round(delta2, 3)))

    delta3 = Oh3 - yx0
    print('    \u03B4\u2083 = {}\n\n'.format(round(delta3, 3)))


if __name__ == '__main__':

    main()
