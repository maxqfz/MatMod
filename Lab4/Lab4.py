# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family='DejaVu Sans')


def EulerIntegrator(h, y0, f):
    """
    Делает один шаг методом Эйлера.
    y0 - начальное значение решения в момент времени t=0,
    h - шаг по времения,
    f(y) - правая часть дифференциального уравнения.
    Возвращает приближенное значение y(h).
    """
    return y0 + h * f(y0)


def oneStepErrorPlot(f, y, integrator):
    """Рисует график зависимости погрешности одного шага
    интегрирования от длины шага.
    f(y) - правая часть дифференциального уравнения,
    y(t) - точное решение,
    integrator(h,y0,f) - аргументы аналогичны EulerIntegrator.
    """
    eps = np.finfo(float).eps
    steps = np.logspace(-10, 0, 50)  # шаги интегрирования
    y0 = y(0)  # начальное значение
    yPrecise = [y(t) for t in steps]  # точные значения решения
    yApproximate = [integrator(t, y0, f) for t in steps]  # приближенные решения
    h = [np.maximum(np.max(np.abs(yp - ya)), eps) for yp, ya in zip(yPrecise, yApproximate)]
    plt.loglog(steps, h, '-')
    plt.xlabel(u"Шаг интегрирования")
    plt.ylabel(u"Погрешность одного шага")


def firstOrderPlot():
    """Рисует на текущем графике прямую y=x."""
    ax = plt.gca()
    steps = np.asarray(ax.get_xlim())
    plt.loglog(steps, steps, '--r')


# Тестовая система.
# Правая часть уравнения y'=f(y).
f = lambda y: y
# Аналитическое решение
yExact = lambda t: np.exp(t)

# Строим график ошибок
oneStepErrorPlot(f, yExact, EulerIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"первый порядок"], loc=2)
plt.show()


def integrate(N, delta, f, y0, integrator):
    """
    Делает N шагов длины delta метода integrator для уравнения y'=f(y) с начальными условиями y0.
    Возвращает значение решения в конце интервала.
    """
    for n in range(N):
        y0 = integrator(delta, y0, f)
    return y0


def intervalErrorPlot(f, y, integrator, T=1, maxNumberOfSteps=1000, numberOfPointsOnPlot=16):
    """
    Рисует график зависимости погрешности интегрирования на интервале
    от длины шага интегрирвания.
    Аргументы повторяют аргументы oneStepErrorPlot.
    """
    eps = np.finfo(float).eps
    numberOfSteps = np.logspace(0, np.log10(maxNumberOfSteps), numberOfPointsOnPlot).astype(np.int)
    steps = T / numberOfSteps  # шаги интегрирования
    y0 = y(0)  # начальное значение
    yPrecise = y(T)  # точнре значения решения на правом конце
    yApproximate = [integrate(N, T / N, f, y0, integrator) for N in numberOfSteps]  # приближенные решения
    h = [np.maximum(np.max(np.abs(yPrecise - ya)), eps) for ya in yApproximate]
    plt.loglog(steps, h, '.-')
    plt.xlabel("Шаг интегрирования")
    plt.ylabel("Погрешность интегрования на интервале")


# Строим график ошибок
intervalErrorPlot(f, yExact, EulerIntegrator)
firstOrderPlot()
plt.legend(["интегратор", "первый порядок"], loc=2)
plt.show()

f = lambda y: 1
yExact = lambda t: t

# Строим график ошибок
oneStepErrorPlot(f, yExact, EulerIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"первый порядок"], loc=2)
plt.title("Метод Эйлера, погрешность одного шага с постоянной производной")
plt.show()


def NewtonIntegrator(h, y0, f):
    """
    Делает один шаг методом Эйлера.
    y0 - начальное значение решения в момент времени t=0,
    h - шаг по времения,
    f(y) - правая часть дифференциального уравнения и его производная.
    Возвращает приближенное значение y(h).
    """
    return y0 + h * f[0](y0) + f[0](y0) * f[1](y0) * h * h / 2


f = (lambda y: y, lambda y: 1)
# Аналитическое решение
yExact = lambda t: np.exp(t)

# Строим график ошибок
oneStepErrorPlot(f[0], yExact, EulerIntegrator)
oneStepErrorPlot(f, yExact, NewtonIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"метод Ньютона", u"первый порядок"], loc=2)
plt.show()


def ModifiedEulerIntegrator(h, y0, f):
    """
    Модифицированный метод Эйлера.
    Аргументы аналогичны EulerIntegrator.
    """
    yIntermediate = y0 + f(y0) * h / 2
    return y0 + h * f(yIntermediate)


f = lambda y: y
yExact = lambda t: np.exp(t)

# Строим график ошибок
oneStepErrorPlot(f, yExact, EulerIntegrator)
oneStepErrorPlot(f, yExact, ModifiedEulerIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"мод. Эйлер", u"первый порядок"], loc=2)
plt.show()


def RungeKuttaIntegrator(h, y0, f):
    """
    Классический метод Рунге-Кутты четвертого порядка.
    Аргументы аналогичны EulerIntegrator.
    """
    k1 = f(y0)
    k2 = f(y0 + k1 * h / 2)
    k3 = f(y0 + k2 * h / 2)
    k4 = f(y0 + k3 * h)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


f = lambda y: y
yExact = lambda t: np.exp(t)

# Строим график ошибок
oneStepErrorPlot(f, yExact, EulerIntegrator)
oneStepErrorPlot(f, yExact, ModifiedEulerIntegrator)
oneStepErrorPlot(f, yExact, RungeKuttaIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"мод. Эйлер", u"метод Рунге-Кутты", u"первый порядок"], loc=2)
plt.show()


def NewtonMethod(F, x0):
    """
    Находит решение уравнения F(x)=0 методом Ньютона.
    x0 - начальное приближение.
    F=(F(x),dF(x)) - функция и ее производная.
    Возвращает решение уравнения.
    """
    for i in range(100):  # ограничиваем максимальное число итераций
        x = x0 - F[0](x0) / F[1](x0)
        if x == x0: break  # достигнута максимальная точность
        x0 = x
    return x0


def BackwardEulerIntegrator(h, y0, f):
    """
    Неявный метод Эйлера.
    Аргументы аналогичны NewtonIntegrator.
    """
    F = (lambda y: y0 + h * f[0](y) - y, lambda y: h * f[1](y) - 1)
    return NewtonMethod(F, y0)


alpha = -10
f = (lambda y: alpha * y, lambda y: alpha)
yExact = lambda t: np.exp(alpha * t)

# Строим график ошибок
oneStepErrorPlot(f[0], yExact, EulerIntegrator)
oneStepErrorPlot(f, yExact, BackwardEulerIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"неявный Эйлер", u"первый порядок"], loc=2)
plt.show()

intervalErrorPlot(f[0], yExact, EulerIntegrator, numberOfPointsOnPlot=32)
intervalErrorPlot(f, yExact, BackwardEulerIntegrator, numberOfPointsOnPlot=16)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"неявный Эйлер", u"первый порядок"], loc=2)
plt.show()

""" ЗАДАНИЕ 2 """
intervalErrorPlot(f[0], yExact, EulerIntegrator)
intervalErrorPlot(f[0], yExact, ModifiedEulerIntegrator)
intervalErrorPlot(f[0], yExact, RungeKuttaIntegrator)
intervalErrorPlot(f, yExact, BackwardEulerIntegrator)
firstOrderPlot()
plt.legend([u"метод Эйлера", u"мод. Эйлер", u"метод Рунге-Кутты", u"неявный Эйлер", u"первый порядок"], loc=2)
plt.show()

""" ЗАДАНИЕ 4 """
def intervalSolutionPlot(f, y0, integrator, stepsCount=100):
    """
    Выводит график - решение уравнения
    f - функция
    y0 - значение функции при t = 0
    integrator - интегратор
    stepsCount - количество шагов
    """
    steps = np.arange(1, stepsCount)
    result = [integrate(N, 1 / stepsCount, f, y0, integrator) for N in steps]
    plt.loglog(result, steps / stepsCount, '.-')
    plt.xlabel("Аргумент")
    plt.ylabel("Решение")

f = lambda y: np.cos(y)
y0 = 1
intervalSolutionPlot(f, y0, EulerIntegrator)
intervalSolutionPlot(f, y0, ModifiedEulerIntegrator)
intervalSolutionPlot(f, y0, RungeKuttaIntegrator)
plt.legend([u"метод Эйлера", u"мод. Эйлер", u"метод Рунге-Кутты"], loc=2)
plt.show()

""" ЗАДАНИЕ 5 """
f = lambda y: (-y[1], y[0])
F = lambda y: np.asarray(f(y), dtype=np.double)
y0 = (0, 1)
intervalSolutionPlot(F, y0, RungeKuttaIntegrator)
plt.legend([u"метод Рунге-Кутты"], loc=2)
plt.show()
