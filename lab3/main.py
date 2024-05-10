import numpy as np
import matplotlib.pyplot as plt

POINTS_AMOUNT = 32

x = np.linspace(0, 2 * np.pi, POINTS_AMOUNT)


def blackman_window(amount):
    window = []
    for i in range(amount):
        element = (0.42 - 0.5 * np.cos((2 * np.pi * i) / amount) + 0.08 * np.cos((4 * np.pi * i) / amount))
        window.append(element)

    return window


def fir_win(filter_len, cut_freq):
    return np.sinc(2 * cut_freq * (np.arange(filter_len) - (filter_len - 1) / 2)) * blackman_window(filter_len)


def get_y():
    return np.cos(2 * x) + np.sin(5 * x)


def get_noise():
    return np.cos(2 * x) + np.sin(5 * x) + np.cos(100 * x)


def get_filtered(y_noise, filter_coefficients):
    return np.convolve(y_noise, filter_coefficients, mode='same')


def main():
    count = 32  # filter len
    cutoff = 5 / count
    filter_coefficients = fir_win(count, cutoff)  # get filter coefficients
    freq_response = np.fft.fft(filter_coefficients)
    plt.figure(figsize=(12, 6))

    y_original = get_y()
    plt.subplot(4, 2, 1)
    plt.plot(x, y_original)
    plt.title('Original function')

    plt.subplot(4, 2, 2)
    plt.plot(np.fft.fft(y_original))
    plt.title('Original function FFT')

    y_noise = get_noise()
    plt.subplot(4, 2, 3)
    plt.plot(x, y_noise)
    plt.title('Function noise')

    plt.subplot(4, 2, 4)
    plt.plot(np.fft.fft(y_noise))
    plt.title('Function noise FFT')

    y_filtered = get_filtered(y_noise, filter_coefficients)
    plt.subplot(4, 2, 5)
    plt.plot(x, y_filtered)
    plt.title('After filtering')

    plt.subplot(4, 2, 6)
    plt.plot(np.fft.fft(y_filtered))
    plt.title('FFT after filtering')

    plt.subplot(4, 2, 8)
    plt.plot(np.abs(freq_response))
    plt.title('Amplitude')


if __name__ == '__main__':
    main()
    plt.tight_layout()
    plt.show()
