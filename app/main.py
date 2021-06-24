# import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numba.typed import List

plt.rcParams.update({'font.size': 12})


@jit(nopython=True)
def resist(x: float, gamma_cr: float, S: float) -> float:
    '''
    Возращает лобовое сопротивление сваи на глубине `x`.

    x -- глубина погружения;
    gamma_cr -- коэффициент условий работы грунта под нижним концом сваи;
    S -- площадь сечения сваи (м^2).
    '''

    # return {
    #         x <= 1: 2900 * 1000 * gamma_cr * S,
    #     1 < x <= 2: 3000 * 1000 * gamma_cr * S,
    #     2 < x <= 3: 3100 * 1000 * gamma_cr * S,
    #     3 < x <= 4: 3200 * 1000 * gamma_cr * S,
    #     4 < x <= 5: 3400 * 1000 * gamma_cr * S,
    #     5 < x <= 6: 3600 * 1000 * gamma_cr * S,
    #     6 < x <= 7: 3700 * 1000 * gamma_cr * S,
    #     7 < x <= 8: 3800 * 1000 * gamma_cr * S,
    #     8 < x <= 9: 3900 * 1000 * gamma_cr * S,
    #    9 < x <= 10: 4000 * 1000 * gamma_cr * S
    # }[True]

    return 6900 * 1000 * gamma_cr * S


@jit(nopython=True)
def xi(x, i, fimp, P, ft, dtm, fi, fls):
    '''
    Считает глубину погружения в момент времени `i`.
    '''
    f = x[i-1] - x[i-2] + ft * dtm + fimp * dtm
    fbs = P * fi * x[i-1]

    if f > 0:
        return x[i-1] + max(max(f - fls * dtm, 0) - fbs * dtm, 0)

    if f + ft + fbs * dtm < 0:
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
    t_table = [0, 6, 12, 18, 23, 27, 34, 40, 44, 55, 61, 64, 72, 77, 82, 86, 90, 99, 105, 113, 120, 125, 135, 150, 158, 185, 203, 230, 263, 276, 285, 291, 310, 320]
    x_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.045, 0.05, 0.08, 0.2, 0.3, 0.55, 0.6, 0.67, 0.8, 0.9, 0.95, 1.05, 1.15, 1.15]
    w_table = [0, 5, 5.16, 5.33, 5.5, 5.6, 5.8, 6, 6.16, 6.33, 6.5, 6.66, 6.83, 7, 7.16, 7.33, 7.5, 9, 9.16, 9.83, 10.5, 11.16, 11.83, 13.83, 14, 14.4, 14.9, 15.4, 16.7, 17.5, 18, 18.5, 19, 19]

    # t_table = [0, 10, 18, 23, 28, 32, 36, 40, 46, 50, 55, 60, 65, 70, 77, 81, 88, 104, 109, 118, 127, 137, 143, 161, 188, 194, 205, 218, 236, 254, 274, 304]
    # x_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.05, 0.2, 0.22, 0.25, 0.28, 0.30, 0.45, 0.6]
    # w_table = [0, 5.00, 5.17, 5.33, 5.50, 5.67, 5.83, 6.00, 6.17, 6.33, 6.50, 6.67, 6.83, 7.00, 7.17, 7.33, 7.50, 9.00, 9.17, 9.83, 10.50, 11.17, 11.83, 13.83, 14.00, 14.43, 14.95, 15.46, 16.75, 17.53, 18.04, 19.07]

    # параметры системы
    g = 9.81
    n = 6  # количество пар дебалансов
    dt = 0.001  # шаг по времени
    dw = 0.01  # шаг по количеству оборотов в секунду
    l = 1.15  # длина сваи (м)
    # l = 0.6  # длина сваи (м)
    P = 0.02 * 4  # периметр сваи (м)
    # P = 0.04 * 4  # периметр сваи (м)
    S = 0.02 * 0.02 - 0.018 * 0.018  # площадь сечения сваи (м^2)
    # S = 0.04 * 0.04 - 0.038 * 0.038  # площадь сечения сваи (м^2)
    M = 37 + (l * 1.2)  # вес машинки + сваи (кг)
    # M = 37 + (l * 2.45)  # вес машинки + сваи (кг)
    gamma_cr = 1.1  # коэффициент условий работы грунта под нижним концом сваи
    gamma_cf = 1.0  # коэффициент условий работы грунта на боковой поверхности
    fi = 17000.0  # расчётное сопротивлене по боковой поверхности (кПа)
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

    noise = np.random.normal(0, 10e-4, n)
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

    curr_t_index = 0
    # пока количество оборотов меньше критического и глубина погружения меньше длины сваи
    while w0 < 50 and x[i - 1] < l:
        noise = np.random.normal(0, 10e-4, n)
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
            if curr_t_index >= len(t_table):
                w.append(w0)
                break
            if t[-1] > t_table[curr_t_index]:
                w0 = w_table[curr_t_index]
                curr_t_index += 1
            # # если за текущую итерацию свая погрузилась меньше, чем на 1 см
            # if abs(x[i] - x[i - period]) <= 0.01:
            #     # увеличиваем обороты погружателя
            #     w0 += 1
        w.append(w0)
        i += 1

        # if x[-1] != 0:
        #     break

    # for i in zip(w, x, all_impulse):
    #     print(i)

    return x, x_table, t, t_table, w, w_table, all_impulse, all_impulse_noise


if __name__ == '__main__':
    x, x_table, t, t_table, w, w_table, all_impulse, all_impulse_noise = main()

    # with open('data.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['x', 't', 'w', 'impulse', 'impulse_noise'])
    #     for data in zip(x, t, w, all_impulse, all_impulse_noise):
    #         writer.writerow(data)

    f, axarr = plt.subplots(3, sharex=True)
    f.subplots_adjust(hspace=0.4)
    axarr[0].plot(t, x, linewidth=4, color='r', label=r'Математическая модель')
    axarr[0].plot(t_table, x_table, linewidth=4, color='m', linestyle='--', label=r'Табличные данные')
    axarr[0].set_title(r'$x(t)$ - глубина погружения (м)')
    axarr[0].set_ylabel(r'$x(t)$ - глубина погружения (м)')
    axarr[0].legend(loc='upper left')
    axarr[1].plot(t, w, linewidth=4, color='b', label=r'Математическая модель')
    axarr[1].plot(t_table, w_table, linewidth=4, color='m', linestyle='--', label=r'Табличные данные')
    axarr[1].set_title(r'$\omega$ - количество оборотов (с)')
    axarr[1].set_ylabel(r'$\omega$ - количество оборотов (с)')
    axarr[1].set_xlabel(r'$t$ - время погружения (с)')
    axarr[1].legend(loc='upper left')
    axarr[2].plot(t, all_impulse_noise, linewidth=4, color='orange', label=r'Импульс с шумом')
    axarr[2].plot(t, all_impulse, linewidth=2, color='g', label=r'Импульс без шума')
    axarr[2].set_title(r'$F$ - сила импульса (Н)')
    axarr[2].set_ylabel(r'$F$ - сила импульса (Н)')
    axarr[2].set_xlabel(r'$t$ - время погружения (с)')
    axarr[2].legend(loc='upper left')
    # axarr[3].plot(t, all_impulse_noise, linewidth=2, color='g')
    # axarr[3].set_title(r'$F$ - сила импульса с шумом (Н)')
    for x in axarr:
        x.grid(True)

    # plt.legend(prop={'size': 16})

    plt.show()
