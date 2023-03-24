"""
ToDo: (https://gitlab.gnome.org/World/OpenPaperwork/libinsane//
"""
from time import sleep
import keyboard
import cv2
import numpy as np
from typing import Tuple
import pyinsane2

# Define constants
DPI = 600
ADJUSTMENT_CONSTANTS = (1.2, 1.0)
QUALITY = 75
THRESHOLD = 80

def capture_scan(device: str) -> np.ndarray:
    """
    Capture a scan using the provided scanner device name.
    """
    pyinsane2.init()
    try:
        device = pyinsane2.Scanner(name=device)
        device.options['resolution'].value = DPI
        device.options['tl-x'].value = int(2.5 * DPI)            # x is across; y is down
        device.options['tl-y'].value = 0
        device.options['br-x'].value = int(8.5 * DPI)
        device.options['br-y'].value = int(8 * DPI)
        scan_session = device.scan(multiple=False)
        # I think this pulls the data out of the buffer into image
        try:
            while True:
                scan_session.scan.read()
        except EOFError:
            pass
        pil_image = scan_session.images[-1]
        image = np.rot90(np.array(pil_image), 3)         # convert to array and rotate 90 CC x 3

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    finally:
        pyinsane2.exit()


def separate_images(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate the input image into ImageA and ImageB.
    """
    mid_point = image.shape[1] // 2
    image_a = image[:, :mid_point]
    image_b = image[:, mid_point:]
    save_images(image_a, image_b, 200, QUALITY)

    return image_a, image_b


def rotate_and_crop(image: np.ndarray) -> np.ndarray:
    """
    Rotate the image to make the objects square and then crop it.

    1. Detects the light-colored rectangular case against the black background by thresholding the image
    2. Finds the largest contour, which should represent the border
    3. Find the bounding rectangle for the contour
    4. Calculate the required padding to ensure the final crop dimensions are correct
    5. Pad the bounding rectangle and crop accordingly.
    """
    # Convert to grayscale and apply threshold to detect the light-colored rectangular border
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('1-thresh1.jpg', thresh, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # Find the largest contour which should represent the slab border
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Find the min area rectangle - output is: mass center(x,y), size(x,y), angle(cw)
    rect = cv2.minAreaRect(contour)
    center = rect[0]
    angle = rect[-1]
    # print(f"minAreaRect: {int(rect[0][0])}, {int(rect[0][1])} : {int(rect[1][0])}, {int(rect[1][1])} : {angle}")

    # Correct the angle  -  angle range is [0, 90)
    if angle > 45:
        angle = angle - 90

    if angle != 0:
        # Rotate the image
        h, w = image.shape[:2]
        # center = (w // 2, h // 2)             # we are using the mass center of minAreaRect
        # print(f"rotating: {angle}")
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotate_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = image     # bypass rotation

    # cv2.imwrite('3-rotated.jpg', rotated, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # recalculate contour of rotated image
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('4-thresh2.jpg', thresh, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # Find the largest contour which should represent the slab border
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Find the bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)          # tl-x, tl-y, w, h

    # w += DPI // 33         # this is a fudge factor to compensate for a consistent bias in the boundingRect output
    # x += DPI // 200

    # print(f"tl-x: {x}; tl-y: {y}; w: {w}; h: {h};")

    # Calculate padding to ensure the final crop dimensions are correct pixels
    target_width = int(3.3 * DPI)
    target_height = int(5.45 * DPI)
    pad_width = (target_width - w) // 2
    pad_height = (target_height - h) // 2
    # print(f"pad_width: {pad_width}; pad_height: {pad_height}")

    # Pad the bounding rectangle while ensuring it stays within image boundaries
    x = max(0, x - pad_width)
    y = max(0, y - pad_height)
    x_end = min(rotated.shape[1], x + target_width)
    y_end = min(rotated.shape[0], y + target_height)
    # print(f"{y}: {y_end}, {x}: {x_end}")

    # Crop the image
    cropped = rotated[y:y_end, x:x_end]

    # cv2.imwrite('6-cropped.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return cropped


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
    scanner_device = '{6BDD1FC6-810F-11D0-BEC7-08002BE2092F}\\0000'
    scan_count = 10

    print("Press '/' to start scanning or 'q' to quit.")

    while True:
        if keyboard.is_pressed('/'):
            # Capture the scan and separate the images
            scan = capture_scan(scanner_device)
            image_a, image_b = separate_images(scan)

            # Rotate, crop, and auto adjust the images
            image_a_processed = auto_adjust(rotate_and_crop(image_a), ADJUSTMENT_CONSTANTS)
            image_b_processed = auto_adjust(rotate_and_crop(image_b), ADJUSTMENT_CONSTANTS)

            # Save the processed images
            save_images(image_a_processed, image_b_processed, scan_count, QUALITY)

            # Increment the scan count
            scan_count += 1
        elif keyboard.is_pressed('q'):
            break
        else:
            sleep(1)


if __name__ == "__main__":
    main()
