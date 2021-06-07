import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numba.typed import List


@jit(nopython=True)
def resist(x: float, gamma_cr: float, S: float) -> float:
    '''
    Возращает лобовое сопротивление сваи на глубине `x`.

    x -- глубина погружения;
    gamma_cr -- коэффициент условий работы грунта под нижним концом сваи;
    S -- площадь сечения сваи (м^2).
    '''

    return {
        x <= 1: 1600 * 1000 * gamma_cr * S,
        1 < x <= 2: 1800 * 1000 * gamma_cr * S,
        2 < x <= 3: 2000 * 1000 * gamma_cr * S,
        3 < x <= 4: 2100 * 1000 * gamma_cr * S,
        4 < x <= 5: 2200 * 1000 * gamma_cr * S,
        5 < x <= 6: 2300 * 1000 * gamma_cr * S,
        6 < x <= 7: 2400 * 1000 * gamma_cr * S,
        7 < x <= 8: 2470 * 1000 * gamma_cr * S,
        8 < x <= 9: 2530 * 1000 * gamma_cr * S,
        9 < x <= 10: 2600 * 1000 * gamma_cr * S
    }[True]


@jit(nopython=True)
def xi(x, i, fimp, P, ft, dtm, fi, fls):
    '''
    Считает глубину погружения в момент времени `i`.
    '''
    f = x[i-1] - x[i-2] + ft * dtm + fimp * dtm
    fbs = P * fi * x[i-1]

    if f > 0:
        return x[i-1] + max(max(f - fls * dtm, 0) - fbs * dtm, 0)

    if f + fbs * dtm < 0:
        print(i)
        print(':(')
        1/0

    return x[i-1] + min(f + fbs * dtm, 0)


@jit(nopython=True)
def sum_(iterable):
    result = 0
    for x in iterable:
        result += x
    return result


@jit(nopython=True)
def main():
    # параметры системы
    g = 9.81
    n = 6  # количество пар дебалансов
    dt = 0.0001  # шаг по времени
    l = 7  # длина сваи (м)
    P = 0.02 * 4  # периметр сваи (м)
    S = 0.02 * 0.02  # площадь сечения сваи (м^2)
    M = 37 + (l * 3.14)  # вес машинки + сваи (кг)
    gamma_cr = 1.1  # коэффициент условий работы грунта под нижним концом сваи
    gamma_cf = 1.0  # коэффициент условий работы грунта на боковой поверхности
    fi = 35000.0  # расчётное сопротивлене по боковой поверхности (кПа)
    # массы дебалансов
    m = [
        2.75758026171761,
        0.969494952543874,
        0.486348994233291,
        0.273755006621712,
        0.155229853500278,
        0.076567059516108
    ]
    # радиусы дебалансов
    R = [
        0.020070401444444,
        0.011900487555556,
        0.008428804666667,
        0.006323725555556,
        0.004761892666667,
        0.003344359555556
    ]

    dtm = dt ** 2 / M
    fls = resist(0, gamma_cr, S)
    ft = M * g

    theta = [0.0] * n
    theta_noise = [0.0] * n

    # инициализируем списки данными первых двух итераций
    x0 = 0
    x1 = max(g * dt ** 2 - fls * dtm, 0)
    x = [x0, x1]  # глубина погружения в каждый момент времени
    t = [0, dt]  # моменты времени
    w0 = 0  # количество оборотов в секунду в текущий момент времени
    w = [w0, w0]  # количество оборотов в секунду в каждый момент времени
    i = 2  # порядковый номер момента времени

    noise = np.random.normal(0, 10e-5, n)
    fimp_0 = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta[k]) for k in range(n)]))
    fimp_noise_0 = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta_noise[k]) for k in range(n)]))
    for k in range(n):
        theta[k] += w0 * (k + 1) * dt * 2 * math.pi
        theta_noise[k] += w0 * (k + 1) * (1 + noise[k]) * dt * 2 * math.pi
    fimp_1 = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta[k]) for k in range(n)]))
    fimp_noise_1 = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta_noise[k]) for k in range(n)]))

    # # лобовое сопротивление в каждый момент времени
    # all_fls = [resist(x0), resist(x1)]
    # # боковое сопротивление в каждый момент времени
    # all_fbs = [0, P * fi * x1]

    # сила импульса в каждый момент времени
    all_impulse = [fimp_0, fimp_1]
    # сила импульса с шумом в каждый момент времени
    all_impulse_noise = [fimp_noise_0, fimp_noise_1]
    # # величина шума в каждый момент времени
    # noise_plot = [0, 0]

    period = int(1 / dt)
    # пока количество оборотов меньше критического и глубина погружения меньше длины сваи
    while w0 < 50 and x[i - 1] < l:
        noise = np.random.normal(0, 10e-5, n)
        for k in range(n):
            theta[k] += w0 * (k + 1) * dt * 2 * math.pi
            theta_noise[k] += w0 * (k + 1) * (1 + noise[k]) * dt * 2 * math.pi
        fimp = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta[k]) for k in range(n)]))
        fimp_noise = sum_(List([m[k] * R[k] * (w0 * (k + 1) * 2 * math.pi) ** 2 * math.cos(theta_noise[k]) for k in range(n)]))
        fls = resist(x[i-1], gamma_cr, S)
        xi_ = xi(x, i, fimp_noise, P, ft, dtm, fi, fls)
        x.append(xi_)
        t.append(dt * i)
        all_impulse.append(fimp)
        all_impulse_noise.append(fimp_noise)
        if not i % period:
            # если за текущую итерацию свая погрузилась меньше, чем на 1 см
            if abs(x[i] - x[i - period]) <= 0.01:
                # увеличиваем обороты погружателя
                w0 += 1
        w.append(w0)
        i += 1

        if x[-1] != 0:
            break

    for i in zip(w, x, all_impulse):
        print(i)

    return x, t, w, all_impulse, all_impulse_noise


if __name__ == '__main__':
    x, t, w, all_impulse, all_impulse_noise = main()

    with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 't', 'w', 'impulse', 'impulse_noise'])
        for data in zip(x, t, w, all_impulse, all_impulse_noise):
            writer.writerow(data)

    f, axarr = plt.subplots(4, sharex=True)
    f.subplots_adjust(hspace=0.4)
    axarr[0].plot(t, x, linewidth=2, color='r')
    axarr[0].set_title(r'$x(t)$ - глубина погружения')
    axarr[1].plot(t, w, linewidth=2, color='b')
    axarr[1].set_title(r'$\omega$ - количество оборотов в секунду')
    axarr[2].plot(t, all_impulse, linewidth=2, color='g')
    axarr[2].set_title(r'$\Sigma$ - импульс')
    axarr[3].plot(t, all_impulse_noise, linewidth=2, color='g')
    axarr[3].set_title(r'$\Sigma$ - импульс с шумом')
    for x in axarr:
        x.grid(True)

    plt.show()
