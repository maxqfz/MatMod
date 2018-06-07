import numpy as np
import matplotlib.pyplot as plt

###
# Элементарные свойства. Редукция аргумента.
###

y=np.linspace(-2,3,100)
x=np.exp(y)
plt.plot(x,y)
plt.xlabel('$x$')
plt.ylabel('$y=\ln x$')
plt.title("График логарифмической функции")
plt.show()

plt.semilogx(x,y)
plt.xlabel('$x$')
plt.ylabel('$y=\ln x$')
plt.title("График логарифма в логарифмической шкале")
plt.show()

x=np.logspace(0,10,100)
y=np.log(x)
plt.semilogx(x,y)
plt.semilogx(1/x,-y)
plt.xlabel('$x$')
plt.ylabel('$y=\ln x$')
plt.title("Редукция аргумента")
plt.show()

x0=np.logspace(-5,5,1000,dtype=np.double)
epsilon=np.finfo(np.double).eps
best_precision=(epsilon/2)*np.abs(1./np.log(x0))
plt.loglog(x0,best_precision, '-k')
plt.loglog(x0,np.full(x0.shape, epsilon), '--r')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\,погрешность$")
plt.legend(["$Минимальная\,погр.$","$Машинная\,погр.$"])
plt.title("Лучшая реализация вычисления логарифма")
plt.show()

###
# Задание 1
###
def getLn1(x):
    return np.log(x ** (1 / 2)) * 2

def getLn2(x):
    return np.log(x / 2) + np.log(2)

smallNumber = 1 - 10 ** -10
print(getLn1(smallNumber))
print(getLn2(smallNumber))
print("Число обусловленности для первого способа: " + str(1 / (2 * np.abs(np.log(np.sqrt(smallNumber))))))
print("Число обусловленности для второго способа: " + str(1 / np.abs(np.log(smallNumber / 2))))

###
# Разложение в степенной ряд
###
def relative_error(x0, x):
    return np.abs(x0-x)/np.abs(x0)

def log_teylor_series(x, N=5):
    a=x-1
    a_k=a # x в степени k. Сначала k=1
    y=a # Значене логарифма, пока для k=1.
    for k in range(2,N): # сумма по степеням
        a_k=-a_k*a # последовательно увеличиваем степень и учитываем множитель со знаком
        y=y+a_k/k
    return y

x=np.logspace(-5,1,1001)
y0=np.log(x)
y=log_teylor_series(x)
plt.loglog(x,relative_error(y0,y),'-k')
plt.loglog(x0,best_precision,'--r')
plt.xlabel('$x$')
plt.ylabel('$(y-y_0)/y_0$')
plt.legend(["$Достигнутая\;погр.$", "$Минимальная\;погр.$"],loc=5)
plt.title("Разложение в степенной ряд")
plt.show()

###
# Задание 2
###

def log_teylor_series_rest(x):
    a=x-1
    a_k=a # x в степени k. Сначала k=1
    y=a # Значене логарифма, пока для k=1.
    #rest = abs((a_k * a))
    N = 2
    # сумма пока остаток больше точности или не наоборот
    while N < 200:
        a_k=-a_k*a # последовательно увеличиваем степень и учитываем множитель со знаком
        y=y+a_k/N
        #rest = abs((a_k * a) / (N + 1))
        N += 1
    return y

y0=np.log(x)
y=log_teylor_series_rest(x)
plt.loglog(x,relative_error(y0,y),'-k')
plt.loglog(x0,best_precision,'--r')
#plt.loglog(x,np.full(x.shape, epsilon), '--b')
plt.loglog()
plt.legend(["$Достигнутая\;погр.$", "$Минимальная\;погр.$"],loc=5)
plt.title("Задание 2")
plt.show()

###
# Аппроксимация многочленами
###
# Узлы итерполяции
N=5
xn=1+1./(1+np.arange(N))
yn=np.log(xn)
# Тестовые точки
x=np.linspace(1+1e-10,2,1000)
y=np.log(x)
# Многочлен лагранжа
import scipy.interpolate
L=scipy.interpolate.lagrange(xn,yn)
yl=L(x)
plt.plot(x,y,'-k')
plt.plot(xn,yn,'.b')
plt.plot(x,yl,'-r')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.title("Аппроксимация многочленами")
plt.show()
plt.semilogy(x,relative_error(y,yl))
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Аппроксимация многочленами, ошибка")
plt.show()

###
# Задание 3
###
N=25
# Узлы итерполяции
un=np.cos(np.pi*(np.arange(N)+1./2)/(N+1))
xn=(1+2*un/3)/(1-2*un/3)
yn=np.log(xn)
# Тестовые точки
x=np.linspace(1/5,5,1000)
y=np.log(x)
u=(3*x-3)/(2*(1+x))
# Многочлен лагранжа
L = scipy.interpolate.lagrange(un, yn)
yl = L(u)
plt.plot(x, y, '-k')
plt.plot(xn, yn, '.b')
plt.plot(x, yl, '-r')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.title("Задание 3")
plt.show()
best_precision = (epsilon/2)*np.abs(1./np.log(x))
plt.semilogy(x, relative_error(y, yl))
plt.loglog(x, best_precision, '--r')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Задание 3, ошибка")
plt.show()

###
# Итерационный метод
###
def log_newton(x, N=15):
    y=1 # начальное приближение
    for j in range(N):
        y=y-1+x/np.exp(y)
    return y

x=np.logspace(-3,3,4000)
y0=np.log(x)
y=log_newton(x)
#best_precision=(epsilon/2)*np.abs(1./np.log(x))
plt.loglog(x,relative_error(y0,y),'-k')
#plt.loglog(x,best_precision,'--r')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Итерационный метод")
plt.show()

###
# Задание 4
###
def log_better_newton(x, N=20):
    y=4 # начальное приближение
    for j in range(N):
        y=y-1+x/np.exp(y)
    return y

x=np.logspace(-3,3,4000)
y0=np.log(x)
y=log_better_newton(x)
best_precision=(epsilon/2)*np.abs(1./np.log(x))
plt.loglog(x,relative_error(y0,y),'-k')
plt.loglog(x,best_precision,'--r')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Задание 4")
plt.show()

###
# Вычисление с помощью таблиц
###
B=4 # число используемых для составления таблицы бит мантиссы
table=np.log((np.arange(0,2**B, dtype=np.double)+0.5)/(2**B))
log2=np.log(2)

def log_table(x):
    M,E=np.frexp(x)
    return log2*E+table[(M*2**B).astype(np.int)]

x=np.logspace(-10,10,1000)
y0=np.log(x)
y=log_table(x)
plt.loglog(x,relative_error(y0,y),'-k')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Вычисление с помощью таблиц")
plt.show()

###
# Задание 5
###
xn=(np.arange(0,2**B, dtype=np.double)+0.5)/(2**B)
yn=np.log(xn)
L=scipy.interpolate.lagrange(xn,yn)

def log_table_lagrange(x):
    M,E=np.frexp(x)
    return log2*E+L(M)

y0=np.log(x)
y=log_table_lagrange(x)
plt.loglog(x,relative_error(y0,y),'-k')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.title("Задание 5")
plt.show()

###
# Задание 6
###
x=np.logspace(-3,3,10000)
y0=np.log(x)
y=log_better_newton(x, 100)
best_precision=(epsilon/2)*np.abs(1./np.log(x))
plt.loglog(x,relative_error(y0,y),'-k')
plt.loglog(x,best_precision,'--r')
plt.loglog(x,np.full(x.shape, epsilon), '--b')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.legend(["","$Минимальная\,погр.$","$Машинная\,погр.$"])
plt.title("Задание 6")
plt.show()