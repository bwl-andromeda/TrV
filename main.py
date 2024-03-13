from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os

print(
    "Здравствуйте! Это лабораторная работа №2 по Теории Вероятности. Выберите действие, которое хотите совершить:"
)


def read_data_from_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Проверяем, не пустая ли строка
                data.append(float(line))
    return data


def plot_correlation_field(X, Y):
    plt.scatter(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Correlation Field")
    plt.grid(True)
    plt.savefig("correlation_field")


def plot_regression_Y_on_X(X, Y, a, b):
    plt.scatter(X, Y, label="Data")
    plt.plot(X, a * np.array(X) + b, color="red", label=f"Y = {a:.2f}X + {b:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regression Y on X")
    plt.legend()
    plt.grid(True)
    plt.savefig("reggression_y_on_x")


def plot_regression_X_on_Y(X, Y, a, b):
    plt.scatter(Y, X, label="Data")
    plt.plot(Y, a * np.array(Y) + b, color="red", label=f"X = {a:.2f}Y + {b:.2f}")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.title("Regression X on Y")
    plt.legend()
    plt.grid(True)
    plt.savefig("reggression_x_on_y")


def regression_Y_on_X():
    X = read_data_from_file("X_data.txt")
    Y = read_data_from_file("Y_data.txt")
    n = len(X)
    sqrX = sum(x * x for x in X)
    x = sum(X)
    xy = sum(x * y for x, y in zip(X, Y))
    y = sum(Y)

    print("Всего дней для Y на X:", n, "\n")
    print("x =", x, "\ny =", y, "\nx^2 =", sqrX, "\nx*y =", xy, "\n")

    print(sqrX, "a +", x, "b =", xy)
    print(x, "a +", n, "b =", y)

    kf = sqrX / x

    x *= kf
    n *= kf
    y *= kf

    b = ((xy - y) - (sqrX - x)) / (x - n)
    a = (y - n) / x

    print("\na =", a, "\nb =", b)

    print("----------------------------")
    if b < 0:
        print("y =", a, "x -", abs(b))
    else:
        print("y =", a, "x +", b)
    print("----------------------------")
    plot_regression_X_on_Y(X, Y, a, b)


def regression_X_on_Y():
    Y = read_data_from_file("X_data.txt")
    X = read_data_from_file("Y_data.txt")
    n = len(X)
    sqrX_2 = sum(x * x for x in X)
    x_2 = sum(X)
    xy_2 = sum(x * y for x, y in zip(X, Y))
    y_2 = sum(Y)

    print("Всего дней для X на Y:", n, "\n")
    print("x =", x_2, "\ny =", y_2, "\ny^2 =", sqrX_2, "\nx*y =", xy_2, "\n")

    print(sqrX_2, "a +", x_2, "b =", xy_2)
    print(x_2, "a +", n, "b =", y_2)

    kf_2 = sqrX_2 / x_2

    x_2 *= kf_2
    n *= kf_2
    y_2 *= kf_2

    b_2 = ((xy_2 - y_2) - (sqrX_2 - x_2)) / (x_2 - n)
    a_2 = (y_2 - n) / x_2

    print("\na =", a_2, "\nb =", b_2)

    print("----------------------------")
    if b_2 < 0:
        print("y =", a_2, "x -", abs(b_2))
    else:
        print("y =", a_2, "x +", b_2)
    print("----------------------------")
    plot_regression_Y_on_X(X, Y, a_2, b_2)


def correlation():
    X = read_data_from_file("X_data.txt")
    Y = read_data_from_file("Y_data.txt")
    n = len(X)
    sqrX = sum(x * x for x in X)
    x = sum(X)
    xy = sum(x * y for x, y in zip(X, Y))
    y = sum(Y)
    sqrY = sum(y * y for y in Y)

    r = (xy * n - x * y) / (sqrt((n * sqrX - x**2) * (n * sqrY - y**2)))
    print("Теснота связи равна:", r, "\n\n")


def calculate_statistics():

    X = read_data_from_file("X_data.txt")
    Y = read_data_from_file("Y_data.txt")

    n = len(X)
    sqrX = sum(x**2 for x in X)
    sqrY = sum(y**2 for y in Y)
    x = sum(X)
    y = sum(Y)
    xy = sum(x * y for x, y in zip(X, Y))

    notx = x / n
    noty = y / n
    notxy = xy / n

    print("Изначальные переменные:")
    print("------------------")
    print("x =", x)
    print("y =", y)
    print("x^2 =", sqrX)
    print("y^2 =", sqrY)
    print("x*y =", xy)
    print("------------------\n")

    K = notxy - notx * noty
    Sx = sqrt(sqrX / n - notx**2)
    Sy = sqrt(sqrY / n - noty**2)
    r = K / (Sx * Sy)
    t = (r * sqrt(n - 2)) / sqrt(1 - r**2)

    print("Значения:")
    print("------------------")
    print("n =", n)
    print("K =", K)
    print("Sx =", Sx)
    print("Sy =", Sy)
    print("r =", r)
    print("t =", t, "\n")

    kf = sqrX / x
    x *= kf
    n *= kf
    y *= kf

    b = ((xy - y) - (sqrX - x)) / (x - n)
    a = (y - n) / x

    y_axb = x_notx = axb_noty = 0

    for xi, yi in zip(X, Y):
        y_axb += (yi - (a * xi + b)) ** 2
        axb_noty += ((a * xi + b) - noty) ** 2
        x_notx += (xi - notx) ** 2

    Sa = sqrt(y_axb / ((n - 2) * x_notx))
    Sb = sqrt(y_axb / ((n - 2) * n))

    Sr2 = axb_noty / (2 - 1)
    S2 = y_axb / (n - 2)
    F = Sr2 / S2

    print("Sa =", Sa)
    print("Sb =", Sb)
    print("------------------\n")

    print("Доверительный интервал:")
    print(f"a: {a - t*Sa} < {a} < {a + t*Sa}")
    print(f"b: {b - t*Sb} < {b} < {b + t*Sb}\n")

    print("Критерий Фишера:")
    print(f"SR^2 = {Sr2}")
    print(f"S^2 = {S2}")
    print(f"F = {F}\n\n")


def graphics():
    X = read_data_from_file("X_data.txt")
    Y = read_data_from_file("Y_data.txt")
    plot_correlation_field(X, Y)
    check = int(
        input(
            "Какой график вы хотите открыть?\n 1 - Корреляционное поле\n2 - X на Y\n3 - Y на X\n"
        )
    )
    if check == 1:
        os.system("xdg-open correlation_field.png")
    elif check == 2:
        os.system("xdg-open reggression_x_on_y.png")
    elif check == 3:
        os.system("xdg-open reggression_y_on_x.png")
    else:
        exit()


menu_options = {
    1: regression_Y_on_X,
    2: regression_X_on_Y,
    3: correlation,
    4: graphics,
    5: calculate_statistics,
    0: exit,
}

while True:
    print(
        "1 - Регрессия Y на X\n2 - Регрессия X на Y\n3 - Теснота связи.\n4 - Графики.\n5 - Корректность.\n0 - Выход из программы\n"
    )
    menu_input = int(input("Ваш выбор: "))
    if menu_input in menu_options:
        menu_options[menu_input]()
    else:
        print("Неправильный выбор, пожалуйста, попробуйте снова.")
