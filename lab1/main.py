# y=sin(2x)+cos(7x) с прореживанием по частоте. график функции, амплитудный спектр бпф, фазовый спектр бпф, обпф
import matplotlib.pyplot as plt
import math
import cmath
import numpy as np

POINTS_AMOUNT = 64


def task_using_libs():
    x = np.linspace(0, 2 * np.pi, POINTS_AMOUNT)
    y = function(x)

    plt.subplot(4, 2, 1)
    plt.plot(x, y)
    plt.title('Function')

    y_fft = np.fft.fft(y)
    # frequency = np.fft.fftfreq(POINTS_AMOUNT)
    amp_spectrum = np.abs(y_fft)
    plt.subplot(4, 2, 3)
    plt.plot(amp_spectrum)
    plt.title('Amplitude spectrum')

    phase_spectrum = np.angle(y_fft)
    plt.subplot(4, 2, 5)
    plt.plot(phase_spectrum)
    plt.title('Phase spectrum')

    inv_y_fft = np.fft.ifft(y_fft)
    plt.subplot(4, 2, 7)
    plt.plot(x, inv_y_fft.real)
    plt.title('Inverse')

    plt.tight_layout()


def function(x):
    return np.sin(2 * x) + np.cos(7 * x)


def get_x_y():
    x = []
    for i in range(POINTS_AMOUNT):
        x.append(2 * math.pi / 64 * i)

    y = []
    for i in range(POINTS_AMOUNT):
        y.append(function(x[i]))

    return x, y


def make_function_graphic(x, y, index, title):
    plt.subplot(4, 2, index)
    plt.plot(x, y)
    plt.title(title)

    plt.tight_layout()


# def fast_fourier_transform(a, n, direction):
#     print(n)
#     if len(a) == 1:
#         return a
#
#     w_n = cmath.cos(2 * cmath.pi / n) + direction * cmath.sqrt(-1) * cmath.sin(2 * cmath.pi / n)
#     w = 1
#     upper_bound = n // 2
#     b = []
#     c = [0 for _ in range(n)]
#     for j in range(upper_bound):
#         b.append(a[j] + a[j + upper_bound])
#         c[j + upper_bound] = ((a[j] - a[j + upper_bound]) * w)
#         w *= w_n
#
#     y_1 = fast_fourier_transform(b, len(b), direction)
#     y_2 = fast_fourier_transform(c, len(c), direction)
#     return y_1 + y_2


def fft(y):
    n = len(y)
    if n <= 1:
        return y
    even = fft(y[0::2])
    odd = fft(y[1::2])
    t = [cmath.exp(-2j * np.pi * k / n) for k in range(n // 2)]
    return [even[k] + t[k] * odd[k] for k in range(n // 2)] + [even[k] - t[k] * odd[k] for k in range(n // 2)]


def ifft(y):
    n = len(y)
    if n <= 1:
        return y
    even = ifft(y[0::2])
    odd = ifft(y[1::2])
    t = [cmath.exp(2j * np.pi * k / n) for k in range(n // 2)]
    return [(even[k] + t[k] * odd[k]) / 2 for k in range(n // 2)] + [(even[k] - t[k] * odd[k]) / 2 for k in
                                                                     range(n // 2)]


def compute_spectra(y_fft):
    amp_spectrum = [abs(x) for x in y_fft]
    # phase_spectrum = [cmath.phase(x) for x in y_fft]
    phase_spectrum = [math.atan2(y.imag, y.real) for y in y_fft]

    return amp_spectrum, phase_spectrum


def main():
    task_using_libs()
    x, y = get_x_y()
    make_function_graphic(x, y, 2, 'Function (manual)')

    # y_fft = fast_fourier_transform(y, len(y), 1)
    # amp_spectrum, phase_spectrum = compute_spectra(y_fft)
    # make_function_graphic(amp_spectrum, 4, 'Amplitude spectrum (manual)')
    # make_function_graphic(x, phase_spectrum, 6, 'Phase spectrum (manual)')

    y_fft = fft(y)
    amp_spectrum, phase_spectrum = compute_spectra(y_fft)
    plt.subplot(4, 2, 4)
    plt.plot(amp_spectrum)
    plt.title('Amplitude')

    plt.subplot(4, 2, 6)
    plt.plot(phase_spectrum)
    plt.title('Phase')

    y = ifft(y_fft)
    plt.subplot(4, 2, 8)
    plt.plot(x, [elem.real for elem in y])
    plt.title('Inverse')

    plt.show()


if __name__ == '__main__':
    main()
