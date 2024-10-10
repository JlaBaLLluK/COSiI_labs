import cv2
import numpy as np

IMAGES_AMOUNT = 11


def process_image(image_number):
    image_path = f"imgs/{image_number}.jpg"
    result_path = f"results/{image_number}.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Unable to load image.")
        return

    scale_percent = 20
    height = int(image.shape[0] * scale_percent / 100)
    width = int(image.shape[1] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # blue colors
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    # blue to white, other to black
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)
    # to gray image with lower brightness
    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_image = (gray_image * 0.02).astype(np.uint8)
    # erode image
    kernel_min = np.ones((5, 5), np.uint8)
    gray_image = cv2.erode(gray_image, kernel_min)
    gray_image = (gray_image * 2).astype(np.uint8)
    # sobel
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    cv2.imwrite(result_path, sobel_combined)


def main():
    for i in range(1, IMAGES_AMOUNT + 1):
        process_image(i)


if __name__ == '__main__':
    main()
