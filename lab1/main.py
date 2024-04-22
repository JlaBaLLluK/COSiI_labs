# y=sin(2x)+cos(7x) с прореживанием по частоте. график функции, амплитудный спектр бпф, фазовый спектр бпф, обпф
import matplotlib.pyplot as plt
import math
import numpy as np

POINTS_AMOUNT = 64
frequency = np.abs(np.fft.fftfreq(POINTS_AMOUNT))


def task_using_libs():
    x = np.linspace(0, 2 * np.pi, POINTS_AMOUNT)
    y = function(x)

    plt.subplot(4, 2, 1)
    plt.plot(x, y)
    plt.title('Function')

    y_fft = np.fft.fft(y)
    amp_spectrum = np.abs(y_fft)
    plt.subplot(4, 2, 3)
    plt.plot(frequency, amp_spectrum)
    plt.title('Amplitude')

    phase_spectrum = np.angle(y_fft)
    plt.subplot(4, 2, 5)
    plt.plot(frequency, phase_spectrum)
    plt.title('Phase')
    
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


def fft(a, direction):
    size = len(a)
    if size == 1:
        return a

    a_even = a[::2]
    a_odd = a[1::2]
    b_even = fft(a_even, direction)
    b_odd = fft(a_odd, direction)
    omega_n = np.exp(-1 * direction * 2j * np.pi / size)
    omega = 1
    y = np.zeros(size, dtype=complex)
    for j in range(size // 2):
        y[j] = b_even[j] + omega * b_odd[j]
        y[j + size // 2] = b_even[j] - omega * b_odd[j]
        omega *= omega_n

    return y


def compute_spectra(y_fft):
    amp_spectrum = [math.sqrt(elem.real ** 2 + elem.imag ** 2) for elem in y_fft]
    phase_spectrum = [math.atan2(elem.imag, elem.real) for elem in y_fft]
    return amp_spectrum, phase_spectrum


def main():
    task_using_libs()

    x, y = get_x_y()
    make_function_graphic(x, y, 2, 'Function (manual)')

    y_fft = fft(y, 1)
    amp_spectrum, phase_spectrum = compute_spectra(y_fft)
    plt.subplot(4, 2, 4)
    plt.plot(frequency, amp_spectrum)
    plt.title('Amplitude (manual)')

    plt.subplot(4, 2, 6)
    plt.plot(frequency, phase_spectrum)
    plt.title('Phase (manual)')

    y = fft(y_fft, -1)
    plt.subplot(4, 2, 8)
    plt.plot(x, [elem.real / 64 for elem in y])
    plt.title('Inverse (manual)')

    plt.show()


if __name__ == '__main__':
    main()
