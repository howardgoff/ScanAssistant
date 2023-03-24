"""
ToDo: (https://gitlab.gnome.org/World/OpenPaperwork/libinsane//
Current error:
  File "C:\Users\Family\py\ScanAssistant\main.py", line 37, in capture_scan
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.7.0) d:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.simd_helpers.hpp:92: error: (-2:Unspecified error) in function '__cdecl cv::impl::`anonymous-namespace'::CvtHelper<struct cv::impl::`anonymous namespace'::Set<1,-1,-1>,struct cv::impl::A0x3c4af206::Set<3,4,-1>,struct cv::impl::A0x3c4af206::Set<0,2,5>,2>::CvtHelper(const class cv::_InputArray &,const class cv::_OutputArray &,int)'
> Invalid number of channels in input image:
>     'VScn::contains(scn)'
> where
>     'scn' is 3

"""

import keyboard
import cv2
import numpy as np
from typing import Tuple
import pyinsane2

# Define constants
DPI = 300
ADJUSTMENT_CONSTANTS = (1.2, 1.0)
QUALITY = 75


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
        device.options['br-y'].value = int(6.5 * DPI)
        scan_session = device.scan(multiple=False)
        # I think this pulls the data out of the buffer into image
        try:
            while True:
                scan_session.scan.read()
        except EOFError:
            pass
        pil_image = scan_session.images[-1]
        image = np.array(pil_image)
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
    scanner_device = '{6BDD1FC6-810F-11D0-BEC7-08002BE2092F}\\0000'
    scan_count = 0

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


if __name__ == "__main__":
    main()
