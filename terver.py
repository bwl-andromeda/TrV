import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Данные
data = {
    "N": np.arange(1, 31),
    "X": np.array(
        [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.9,
            1.3,
            1.8,
            1.3,
            0.9,
            0.9,
            0.9,
            0.9,
            1.3,
            1.8,
            2.7,
            2.2,
            2.7,
            2.2,
            1.8,
            1.8,
            1.8,
            1.3,
            1.8,
            0.9,
            1.3,
            1.3,
            0.9,
            0.9,
            0.9,
        ]
    ),
    "Y": np.array(
        [
            0,
            0.3,
            0.7,
            0.8,
            0.8,
            1.1,
            1.9,
            1.9,
            2.3,
            2.7,
            2.7,
            2.5,
            1.9,
            1.8,
            2.1,
            1.3,
            1.1,
            0.9,
            0.7,
            0.7,
            2.3,
            2.7,
            2.9,
            3,
            3.1,
            3.1,
            2.9,
            2.7,
            2.4,
            2,
        ]
    ),
}

# Линейная регрессия Y на X
slope_yx, intercept_yx, r_value_yx, p_value_yx, std_err_yx = linregress(
    data["X"], data["Y"]
)
regression_equation_yx = f"Y = {slope_yx:.2f}X + {intercept_yx:.2f}"
print("Уравнение линейной регрессии Y на X:", regression_equation_yx)

# Линейная регрессия X на Y
slope_xy, intercept_xy, r_value_xy, p_value_xy, std_err_xy = linregress(
    data["Y"], data["X"]
)
regression_equation_xy = f"X = {slope_xy:.2f}Y + {intercept_xy:.2f}"
print("Уравнение линейной регрессии X на Y:", regression_equation_xy)

# Коэффициент корреляции
correlation_coefficient = np.corrcoef(data["X"], data["Y"])[0, 1]
print("Коэффициент корреляции:", correlation_coefficient)

# Уравнение выборочной регрессии Y на X
regression_equation_yx_sample = f"Y = {correlation_coefficient:.2f}X + {np.mean(data['Y']) - correlation_coefficient*np.mean(data['X']):.2f}"
print("Уравнение выборочной регрессии Y на X:", regression_equation_yx_sample)

# Уравнение выборочной регрессии X на Y
regression_equation_xy_sample = f"X = {correlation_coefficient:.2f}Y + {np.mean(data['X']) - correlation_coefficient*np.mean(data['Y']):.2f}"
print("Уравнение выборочной регрессии X на Y:", regression_equation_xy_sample)

# Корреляционное поле
plt.scatter(data["X"], data["Y"])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Корреляционное поле")
plt.grid(True)
plt.savefig("graphics/correlation_field.png")
print(
    'График был сохранен в той же директории с исполняемым файлом под названием: "correlation_field"'
)
# График регрессии Y на X
plt.scatter(data["X"], data["Y"], label="Исходные данные")
plt.plot(
    data["X"],
    slope_yx * data["X"] + intercept_yx,
    color="red",
    label="Линейная регрессия Y на X",
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Регрессия Y на X")
plt.legend()
plt.grid(True)
plt.savefig("graphics/regression_yx.png")
print(
    'График был сохранен в той же директории с исполняемым файлом под названием "regression_yx"'
)
# График регрессии X на Y
plt.scatter(data["Y"], data["X"], label="Исходные данные")
plt.plot(
    data["Y"],
    slope_xy * data["Y"] + intercept_xy,
    color="red",
    label="Линейная регрессия X на Y",
)
plt.xlabel("Y")
plt.ylabel("X")
plt.title("Регрессия X на Y")
plt.legend()
plt.grid(True)
plt.savefig("graphics/regression_xy.png")
print(
    'График был сохранен в той же директории с исполняемым файлом под названием "regression_xy"'
)
# Проверка значимости уравнения регрессии Y на X по критерию Фишера-Снедекера
n = len(data["X"])
k = 2  # количество параметров модели (наклон и пересечение)
F_value = (r_value_yx**2 / (1 - r_value_yx**2)) * (n - k - 1) / k
alpha = 0.05  # уровень значимости
critical_value = 3.38  # критическое значение для alpha=0.05 и k1=2, k2=28
if F_value > critical_value:
    print("Уравнение регрессии Y на X является значимым.")
else:
    print("Уравнение регрессии Y на X не является значимым.")

# Интервальная оценка коэффициентов a и b с доверительной вероятностью gamma=0.95
t_value = 2.048  # значение t-статистики для alpha=0.05 и 28 степеней свободы
SE_a = std_err_yx * np.sqrt(
    1 / n + np.mean(data["X"]) ** 2 / ((n - 1) * np.var(data["X"]))
)
SE_b = std_err_yx / np.sqrt((n - 1) * np.var(data["X"]))
a_lower = slope_yx - t_value * SE_a
a_upper = slope_yx + t_value * SE_a
b_lower = intercept_yx - t_value * SE_b
b_upper = intercept_yx + t_value * SE_b
print(f"Интервальная оценка коэффициента a: ({a_lower:.2f}, {a_upper:.2f})")
print(f"Интервальная оценка коэффициента b: ({b_lower:.2f}, {b_upper:.2f})")

# Проверка значимости выборочного коэффициента корреляции
t_value_corr = correlation_coefficient * np.sqrt(
    (n - 2) / (1 - correlation_coefficient**2)
)
t_critical_corr = 2.048  # для alpha=0.05 и 28 степеней свободы
if abs(t_value_corr) > t_critical_corr:
    print("Выборочный коэффициент корреляции значим.")
else:
    print("Выборочный коэффициент корреляции не значим.")
