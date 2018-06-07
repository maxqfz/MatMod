import numpy as np
import math

base=10 # параметр, может принимать любые целые значения > 1

def exact_sum(K):
    "Точное значение суммы всех элементов."
    return 1.

def samples(K):
    "Элементы выборки."
    # создаем K частей из base^k одинаковых значений
    parts = [np.full((base**k,), float(base)**(-k)/K) for k in range(0, K)]
    # создаем выборку объединяя части
    samples = np.concatenate(parts)
    # перемешиваем элементы выборки и возвращаем
    return np.random.permutation(samples)

def samples2(K):
    parts = [np.full((base**k,), float(base)**(-k)/K) for k in range(0, K)]
    parts += [np.full((base**k,), float(base)**(-k)/K) for k in range(0, K)]
    parts += [np.full((base**k,), -float(base)**(-k)/K) for k in range(0, K)]
    samples = np.concatenate(parts)
    return np.random.permutation(samples)

def samplessin(K):
    parts = [np.full((base**k), math.sin(k)) for k in range(0, K)]
    parts += [np.full((base**k), math.sin(k)) for k in range(0, K)]
    parts += [np.full((base**k), math.sin(k)) for k in range(0, K)]
    samples = np.concatenate(parts)
    return np.random.permutation(samples)

def direct_sum(x):
    "Последовательная сумма всех элементов вектора x"
    s = 0.
    for e in x: 
        s += e
    return s

def number_of_samples(K):
    "Число элементов в выборке"
    return np.sum([base**k for k in range(0, K)])

def exact_mean(K):
    "Значение среднего арифметического по выборке с близкой к машинной точностью."
    return 1./number_of_samples(K)

def exact_variance(K):
    "Значение оценки дисперсии с близкой к машинной точностью."
    # разные значения элементов выборки
    values = np.asarray([float(base)**(-k)/K for k in range(0, K)], dtype=np.double)
    # сколько раз значение встречается в выборке
    count = np.asarray([base**k for k in range(0, K)])
    return np.sum(count*(values-exact_mean(K))**2)/number_of_samples(K)

K=7 # число слагаемых
x = samples(K) # сохраняем выборку в массив
print("Число элементов:", len(x))
print("Самое маленькое и большое значения:", np.min(x), np.max(x))

exact_sum_for_x=exact_sum(K) # значение суммы с близкой к машинной погрешностью
direct_sum_for_x=direct_sum(x) # сумма всех элементов по порядку

def relative_error(x0, x):
    "Погрешность x при точном значении x0"
    return np.abs(x0-x)/np.abs(x)

print("Погрешность прямого суммирования:", relative_error(exact_sum_for_x, direct_sum_for_x))

sorted_x=x[np.argsort(x)]
sorted_sum_for_x=direct_sum(sorted_x)
print("Погрешность суммирования по возрастанию:", relative_error(exact_sum_for_x, sorted_sum_for_x))

sorted_x=x[np.argsort(x)[::-1]]
sorted_sum_for_x=direct_sum(sorted_x)
print("Погрешность суммирования по убыванию:", relative_error(exact_sum_for_x, sorted_sum_for_x))

sorted_x=sorted(x, key=lambda item: abs(item))
sorted_sum_for_x=direct_sum(sorted_x)
print("Погрешность суммирования по возрастанию модуля:", relative_error(exact_sum_for_x, sorted_sum_for_x))

def Kahan_sum(x):
    s=0.0 # частичная сумма
    c=0.0 # сумма погрешностей
    for i in x:
        y=i-c      # первоначально y равно следующему элементу последовательности
        t=s+y      # сумма s может быть велика, поэтому младшие биты y будут потеряны
        c=(t-s)-y  # (t-s) отбрасывает старшие биты, вычитание y восстанавливает младшие биты
        s=t        # новое значение старших битов суммы
    return s

Kahan_sum_for_x=Kahan_sum(x) # сумма всех элементов по порядку
print("Погрешность суммирования по Кэхэну:", relative_error(exact_sum_for_x, Kahan_sum_for_x))

