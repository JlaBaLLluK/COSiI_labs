import numpy as np
import matplotlib.pyplot as plt

POINTS_AMOUNT = 8
plt.figure(figsize=(12, 8))


def make_functions_graphics(x, y, z):
    plt.subplot(3, 3, 1).set_title('y = cos(2x)')
    plt.plot(x, y)
    plt.subplot(3, 3, 4).set_title('z = sin(5x)')
    plt.plot(x, z)


def linear_convolution(y, z):
    y_len = len(y)
    z_len = len(z)
    result = []
    for n in range(0, y_len + z_len - 1):
        result_elem = 0
        for m in range(0, n):
            a = 0
            if 0 <= m < y_len:
                a = y[m]

            b = 0
            if 0 <= (n - m) < z_len:
                b = z[(n - m)]

            result_elem += a * b

        result.append(result_elem)

    return result


def cyclic_convolution(y, z):
    y_len = len(y)
    result = []
    for n in range(0, y_len):
        result_elem = 0
        for m in range(0, y_len):
            result_elem += y[m] * z[n - m]

        result.append(result_elem / y_len)

    return result


def correlation(y, z):
    y_len = len(y)
    result = []
    for n in range(0, y_len):
        result_elem = 0
        for m in range(0, y_len):
            a = y[m]
            if (n + m) >= y_len:
                b = z[(n - m) % y_len]
            else:
                b = z[(n - m)]

            result_elem += a * b

        result.append(result_elem / y_len)

    return result


def fft_result(y, z):
    y_fft = np.fft.fft(y)
    z_fft = np.fft.fft(z)
    mult_result = [y_fft[i].real * z_fft[i].real for i in range(len(y_fft))]
    return np.fft.ifft(mult_result)


def make_linear_convolution_correlation_graphics(y, z, linear_convolution_result):
    plt.subplot(3, 3, 2).set_title('Linear convolution (manual)')
    plt.plot(linear_convolution_result)

    linear_convolution_result = np.convolve(y, z)
    plt.subplot(3, 3, 5).set_title('Linear convolution (lib)')
    plt.plot(linear_convolution_result)

    correlation_result = np.correlate(y, z[::-1], 'full')
    plt.subplot(3, 3, 8).set_title('Correlation (lib)')
    plt.plot(correlation_result)


def make_cyclic_convolution_correlation_fft_graphics(cyclic_convolution_result, correlation_result, fft_res):
    plt.subplot(3, 3, 3).set_title('Cyclic convolution (manual)')
    plt.plot(cyclic_convolution_result)
    plt.subplot(3, 3, 6).set_title('Correlation (manual)')
    plt.plot(correlation_result)
    plt.subplot(3, 3, 9).set_title('FFT')
    plt.plot(fft_res)


def main():
    x = np.linspace(0, 2 * np.pi, POINTS_AMOUNT)
    y = np.cos(2 * x)
    z = np.sin(5 * x)
    make_functions_graphics(x, y, z)
    linear_convolution_result = linear_convolution(y, z)
    make_linear_convolution_correlation_graphics(y, z, linear_convolution_result)
    cyclic_convolution_result = cyclic_convolution(y, z)
    correlation_result = correlation(y, z)
    fft_res = fft_result(y, z)
    make_cyclic_convolution_correlation_fft_graphics(cyclic_convolution_result, correlation_result, fft_res)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
