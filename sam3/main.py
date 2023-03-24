`from typing import List, Tuple
import cv2
import numpy as np
import pytesseract
import twain

ScanCount = 0


def scan_images() -> Tuple[np.ndarray, np.ndarray]:
    """Uses the twain driver to scan the top 7 inches of the scanner
    and returns the two rectangle shaped items measuring 3.25 inches
    by 5.50 inches each on a black background as separate images."""

    with twain.SourceManager() as sm:
        # get the first available scanner
        devices = sm.GetSourceList()
        if not devices:
            raise Exception("No TWAIN devices found")

        scanner = sm.OpenSource(devices[0])

        # configure scanner settings
        scanner.SetCapability(twain.ICAP_XRESOLUTION, twain.TWTY_FIX32, 600)
        scanner.SetCapability(twain.ICAP_YRESOLUTION, twain.TWTY_FIX32, 600)
        scanner.SetCapability(twain.ICAP_PIXELTYPE, twain.TWTY_UINT16, twain.TWPT_BW)
        scanner.SetCapability(twain.ICAP_AUTODISCARDBLANKPAGES, twain.TWTY_BOOL, True)

        # start the scan process
        scanner.RequestAcquire(0, 0)
        scanner.XferImageNatively(twain.TWSX_NATIVE)

        # retrieve scanned image data
        handle = scanner.GetImage(twain.TWDX_NATIVEPOINTER)
        image_data = np.frombuffer(handle, dtype=np.uint8)

        # convert image data to grayscale and resize to 8.5x11 inches
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        target_height = int(11 * 600)  # convert inches to pixels at 600dpi
        target_width = int(8.5 * 600)
        image = cv2.resize(image, (target_width, target_height))

        # find the locations of the two rectangles
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        # extract the two rectangle shaped items
        x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[1])
        image_a = image[y1:y1 + h1, x1:x1 + w1]
        image_b = image[y2:y2 + h2, x2:x2 + w2]

        return image_a, image_b


def rotate_image(image: np.ndarray) -> np.ndarray:
    """Rotates the image so that it is square by detecting the edges
    of the scanned object."""

    # find the edges of the object
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = [np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]]

    # calculate the median angle of the edges
`