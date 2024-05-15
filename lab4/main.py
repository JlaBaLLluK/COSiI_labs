import numpy as np
import matplotlib.pyplot as plt

POINTS_AMOUNT = 64


def haar_wavelet_transform(signal, level):
    coefficients_high = []
    coefficients_low = None
    for _ in range(level):
        length = len(signal)
        if length == 1:
            coefficients_high.append(signal)
            break

        half_length = length // 2
        low_pass = np.zeros(half_length)
        high_pass = np.zeros(half_length)

        for i in range(half_length):
            low_pass[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
            high_pass[i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)

        coefficients_high.append(high_pass)
        signal = low_pass

    coefficients_low = signal
    coefficients_high.reverse()
    return coefficients_high, coefficients_low


def inverse_haar_wavelet_transform(coefficients_H, coefficients_L):
    level = len(coefficients_H)
    signal = coefficients_L
    for i in range(level):
        length = len(coefficients_H[i])
        reconstructed_signal = np.zeros(length * 2)
        for j in range(length):
            reconstructed_signal[2 * j] = (signal[j] + coefficients_H[i][j]) / np.sqrt(2)
            reconstructed_signal[2 * j + 1] = (signal[j] - coefficients_H[i][j]) / np.sqrt(2)
        signal = reconstructed_signal
    return signal


def main():
    x = np.linspace(0, 2 * np.pi, POINTS_AMOUNT)
    y = np.cos(x) + np.sin(x)

    level = int(np.log2(POINTS_AMOUNT))

    coefficients_high, coefficient_low = haar_wavelet_transform(y, level)
    reconstructed_signal = inverse_haar_wavelet_transform(coefficients_high, coefficient_low)

    print("H coefficients:")
    for i, coefficient in enumerate(coefficients_high):
        print(f"Level {i + 1}: {coefficient}")

    print("Last L coefficient:", coefficient_low)

    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.title('Signal')

    plt.subplot(2, 1, 2)
    plt.plot(x, reconstructed_signal)
    plt.title('Reconstructed signal')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
