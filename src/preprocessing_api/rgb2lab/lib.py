import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def img_to_lab(image_path: str, WIDTH = None, HEIGHT = None):
    """
    Reads an image from a file path, optionally resizes it while maintaining aspect ratio,
    and converts it from BGR to CIELAB color space.

    Args:
        image_path (str): File system path to the input image.
        WIDTH (optional): Target display width for aspect-ratio resizing. Requires HEIGHT.
        HEIGHT (optional): Target display height for aspect-ratio resizing. Requires WIDTH.

    Returns:
       Image matrix converted to the CIELAB color space.

    Raises:
        ValueError: If no image is found at the path, or if only one resize dimension is supplied.
    """
    img_bgr = cv2.imread(image_path)

    # Raise error if image failed to load
    if img_bgr is None:
        raise ValueError(f"No image found at path: {image_path}")

    # Validate that both dimensions are supplied if resizing
    if WIDTH is None and HEIGHT is not None:
        raise ValueError("Give both width and height to crop image.")
    elif HEIGHT is None and WIDTH is not None:
        raise ValueError("Give both width and height to crop image.")

    # Crop/resize image while preserving aspect ratio if dimensions are provided
    elif WIDTH is not None and HEIGHT is not None:
        h, w = img_bgr.shape[:2]
        if h < w:
            img_bgr = cv2.resize(img_bgr, (WIDTH, math.floor(h / (w / HEIGHT))))
        else:
            img_bgr = cv2.resize(img_bgr, (math.floor(w / (h / HEIGHT)), HEIGHT))

    # Convert BGR image matrix to CIELAB space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    return img_lab


def get_lab_channels_and_gm(img_lab: np.ndarray):
    """
    Splits a CIELAB image matrix into its individual channels and calculates the 
    Sobel gradient magnitude across the L (Lightness) channel for texture/edge detection.

    Args:
        img_lab (np.ndarray): Input image matrix in CIELAB color space.

    Returns:
        tuple: (L_channel, a_channel, b_channel, gradient_magnitude)
            - L_channel (np.ndarray): Lightness channel (0=Black, 255=White).
            - a_channel (np.ndarray): Green-to-Red axis.
            - b_channel (np.ndarray): Blue-to-Yellow axis.
            - gradient_magnitude (np.ndarray): Edge/texture intensity map derived from Sobel filters.
    """
    # Split the 3-channel LAB matrix into single 2D arrays
    L, a, b = cv2.split(img_lab)

    # Calculate 3x3 Sobel directional derivatives along X and Y on Lightness channel
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)

    # Square the directional gradients
    sobelx_sq = np.square(sobelx)
    sobely_sq = np.square(sobely)

    # Compute total gradient magnitude: sqrt(sobelx^2 + sobely^2)
    gradient_magnitude = np.sqrt(sobelx_sq + sobely_sq)

    return L, a, b, gradient_magnitude

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def check_dir(directory_path: str, depth_scanner: bool = False, save_rgb: bool = False):
    """
    Scans a target directory for valid image files, creates an individual output folder per image 
    under 'processed_output', converts and saves the LAB image, and optionally extracts channel heatmaps 
    and gradient maps.

    Args:
        directory_path (str): File system path to the directory containing input images.
        depth_scanner (bool, optional): If True, calculates L, A, B, and Sobel gradient heatmaps 
            and saves them into a nested 'channel_dir' subfolder. Defaults to False.
        save_rgb (bool, optional): If True, restores and saves a copy of the original color image 
            alongside the LAB output. Defaults to False.
    """
    out_root = os.path.join(directory_path, "processed_output")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip subdirectories
        if not os.path.isfile(file_path):
            continue

        # Ignore files with unsupported extensions
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_EXTS:
            continue

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            continue

        # 1. Create dedicated folder for this specific image (e.g. processed_output/image_01/)
        img_dir = os.path.join(out_root, stem)
        os.makedirs(img_dir, exist_ok=True)

        # 2. ALWAYS convert to LAB and save the base LAB image in the image's folder
        img_lab = img_to_lab(file_path)
        cv2.imwrite(os.path.join(img_dir, f"{stem}_lab.png"), img_lab)

        # Optionally convert LAB back to RGB and save original color image
        if save_rgb:
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            cv2.imwrite(os.path.join(img_dir, f"{stem}_rgb.png"), img_rgb)

        # 3. IF depth_scanner is True, extract channel heatmaps into 'channel_dir'
        if depth_scanner:
            outdir = os.path.join(img_dir, "channel_dir")
            os.makedirs(outdir, exist_ok=True)

            L, a, b, gradient_magnitude = get_lab_channels_and_gm(img_lab)
            iter_list = [L, a, b, gradient_magnitude]

            for ind, item in enumerate(iter_list):
                match ind:
                    case 0:
                        channel_name = "L"
                        cmap = "gray"
                    case 1:
                        channel_name = "A"
                        cmap = "coolwarm"
                    case 2:
                        channel_name = "B"
                        cmap = "coolwarm"
                    case 3:
                        channel_name = "gradient"
                        cmap = "magma"

                # Save heatmapped channels into 'outdir' (channel_dir)
                item_path = os.path.join(outdir, f"{stem}_{channel_name}_channel.png")
                plt.imsave(item_path, item, cmap=cmap)