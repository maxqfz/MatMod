import numpy as np

def direct_sum(x):
    """Последовательная сумма всех элементов вектора x"""
    s = 0.
    for e in x:
        s += e
    return s

def Kahan_sum(x):
    s=0.0 # частичная сумма
    c=0.0 # сумма погрешностей
    for i in x:
        y=i-c      # первоначально y равно следующему элементу последовательности
        t=s+y      # сумма s может быть велика, поэтому младшие биты y будут потеряны
        c=(t-s)-y  # (t-s) отбрасывает старшие биты, вычитание y восстанавливает младшие биты
        s=t        # новое значение старших битов суммы
    return s

def relative_error(x0, x):
    """Погрешность x при точном значении x0"""
    return np.abs(x0-x)/np.abs(x)

# параметры выборки
mean=1e6 # среднее
delta=1e-5 # величина отклонения от среднего

def samples(N_over_two):
    """Генерирует выборку из 2*N_over_two значений с данным средним и среднеквадратическим
    отклонением."""
    x=np.full((2*N_over_two,), mean, dtype=np.double)
    x[:N_over_two]+=delta
    x[N_over_two:]-=delta
    return np.random.permutation(x)

def exact_mean():
    """Значение среднего арифметического по выборке с близкой к машинной точностью."""
    return mean

def exact_variance():
    """Значение оценки дисперсии с близкой к машинной точностью."""
    return delta**2

x=samples(1000000)

print("Размер выборки:", len(x))
print("Среднее значение:", exact_mean())
print("Оценка дисперсии:", exact_variance())
print("Ошибка среднего для встроенной функции:",relative_error(exact_mean(),np.mean(x)))
print("Ошибка дисперсии для встроенной функции:",relative_error(exact_variance(),np.var(x)))

def direct_mean(x):
    """Среднее через последовательное суммирование."""
    return direct_sum(x)/len(x)

#print("Ошибка среднего для последовательного суммирования:",relative_error(exact_mean(),direct_mean(x)))

def direct_second_var(x):
    """Вторая оценка дисперсии через последовательное суммирование."""
    return direct_mean(x**2)-direct_mean(x)**2

def online_second_var(x):
    """Вторая оценка дисперсии через один проход по выборке"""
    m=x[0] # накопленное среднее
    m2=x[0]**2 # накопленное среднее квадратов
    for n in range(1,len(x)):
        m=(m*(n-1)+x[n])/n
        m2=(m2*(n-1)+x[n]**2)/n
    return m2-m**2

#print("Ошибка второй оценки дисперсии для последовательного суммирования:",relative_error(exact_variance(),direct_second_var(x)))
#print("Ошибка второй оценки дисперсии для однопроходного суммирования:",relative_error(exact_variance(),online_second_var(x)))

def direct_first_var(x):
    """Первая оценка дисперсии через последовательное суммирование."""
    return direct_mean((x-direct_mean(x))**2)

#print("Ошибка первой оценки дисперсии для последовательного суммирования:",relative_error(exact_variance(),direct_first_var(x)))

def Kahan_mean(x):
    """Среднее через суммирование Кэхена."""
    return Kahan_sum(x)/len(x)

def Kahan_first_var(x):
    """Первая оценка дисперсии через последовательное суммирование."""
    return Kahan_mean((x - Kahan_mean(x)) ** 2)

#print("Ошибка первой оценки дисперсии для суммирования Кэхена:",relative_error(exact_variance(),Kahan_first_var(x)))

def online_first_var(x):
    n = len(x)
    #Не можем рассчитать менее чем для трёх элементов
    if n <= 2:
        return
    E = (x[0] + x[1]) / 2
    firstvar = 0
    for i in range(2, len(x)):
        Eprev = E
        E = ((i - 1) * E + x[i]) / i
        correction = ((i - 1) / i) * ((x[i] - Eprev) ** 2 / i)
        firstvar = ((i - 2) * firstvar + (x[i] - E) ** 2 + correction) / (i - 1)
    return firstvar

print("Первая оценка дисперсии через последовательное суммирование: ", direct_first_var(x))
print("Первая оценка дисперсии через однопроходное суммирование: ", online_first_var(x))