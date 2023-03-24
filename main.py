"""
pip install pyinsane2
# pip install pytwain
pip install numpy
pip install opencv-python-headless
pip install keyboard
"""

import os
import keyboard
import cv2
import numpy as np
from typing import Tuple, List
# from twain import Scanner
import pyinsane2

# Define constants
DPI = 600
ADJUSTMENT_CONSTANTS = (1.2, 1.0)
QUALITY = 75


# def capture_scan(scanner: Scanner) -> np.ndarray:
#     """
#     Capture a scan using the provided scanner.
#     """
#     scanner.resolution = DPI
#     scanner.set_region(0, 0, 7 * DPI, 3.25 * 2 * DPI)
#     return cv2.imdecode(np.frombuffer(scanner.capture(), dtype=np.uint8), cv2.IMREAD_COLOR)


def capture_scan(device: str) -> np.ndarray:
    """
    Capture a scan using the provided scanner device name.
    """
    pyinsane2.init()
    try:
        device = pyinsane2.Scanner(name=device)
        device.options['resolution'].value = DPI
        device.options['tl-x'].value = 0
        device.options['tl-y'].value = 0
        # device.options['br-x'].value = 7 * DPI
        device.options['br-y'].value = int(3.25 * DPI)
        scan_session = device.scan(multiple=False)
        image = None
        for chunk in scan_session:
            if image is None:
                image = np.array(chunk, dtype=np.uint8)
            else:
                image = np.vstack((image, np.array(chunk, dtype=np.uint8)))
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    finally:
        pyinsane2.exit()


def separate_images(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate the input image into ImageA and ImageB.
    """
    mid_point = image.shape[1] // 2
    image_a = image[:, :mid_point]
    image_b = image[:, mid_point:]
    return image_a, image_b


def rotate_and_crop(image: np.ndarray) -> np.ndarray:
    """
    Rotate the image to make the objects square and then crop it.
    """
    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find the contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Find the rotated rectangle and its angle
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]

    # Correct the angle
    if angle < -45:
        angle = 90 + angle

    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Crop the image
    x, y, w, h = cv2.boundingRect(contour)
    cropped = rotated[y:y + h, x:x + w]

    # Resize the image to the final dimensions
    resized = cv2.resize(cropped, (2100, 3450), interpolation=cv2.INTER_AREA)

    return resized


def auto_adjust(image: np.ndarray, adjustment_constants: Tuple[float, float]) -> np.ndarray:
    """
    Auto adjust color and brightness.
    """
    # """
    # Auto-adjust the color and brightness of the image using the provided adjustment constants.
    # """
    # return cv2.convertScaleAbs(image, alpha=adjustment_constants[0], beta=adjustment_constants[1])

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=adjustment_constants[0], tileGridSize=(8, 8))
    l_adjusted = clahe.apply(l)
    l_adjusted = cv2.addWeighted(l, adjustment_constants[1], l_adjusted, 1 - adjustment_constants[1], 0)
    image_lab_adjusted = cv2.merge((l_adjusted, a, b))
    image_adjusted = cv2.cvtColor(image_lab_adjusted, cv2.COLOR_Lab2BGR)
    return image_adjusted


def save_images(image_a: np.ndarray, image_b: np.ndarray, scan_count: int, quality: int):
    """
    Save ImageA and ImageB as jpg files with the specified quality.
    """
    file_a = f"Inv{scan_count:05d}.jpg"
    file_b = f"Inv{(scan_count - 1):05d}b.jpg"

    cv2.imwrite(file_a, image_a, [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(file_b, image_b, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main():
    # Initialize the scanner
    # scanner = Scanner()
    # scanner_device = "Epson Perfection V700/V750"
    # scanner_device = 'EPSON Perfection V700/V750'
    scanner_device = '{6BDD1FC6-810F-11D0-BEC7-08002BE2092F}\\0000'
    scan_count = 0

    scan = capture_scan(scanner_device)
    print(type(scan))
    while False:
        if keyboard.is_pressed('/'):
            # Capture the scan and separate the images
            # scan = capture_scan(scanner)

            scan = capture_scan(scanner_device)
            image_a, image_b = separate_images(scan)

            # Rotate, crop, and auto adjust the images
            image_a_processed = auto_adjust(rotate_and_crop(image_a), ADJUSTMENT_CONSTANTS)
            image_b_processed = auto_adjust(rotate_and_crop(image_b), ADJUSTMENT_CONSTANTS)

            # Save the processed images
            save_images(image_a_processed, image_b_processed, scan_count, QUALITY)

            # Increment the scan count
            scan_count += 1


if __name__ == "__main__":
    main()

    # pyinsane2.init()
    # devs = pyinsane2.get_devices()
    # target_dev = None
    #
    # for dev in devs:
    #     if dev.name == "hplip://32.13241":
    #         target_dev = dev
    #         break
    #
    # if target_dev is None:
    #     print("Not found")
    # else:
    #     print("Will use {} {}".format(target_dev.vendor, target_dev.model))
