import numpy as np
import cv2
import time
import os


def compute_energy(image):
    """
    Compute the energy map of the image using sum of gradients
    Args:
        image: input image
    Returns:
        energy: array of the energy map
    """

    if image.ndim == 3:  # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)

    height, width = gray.shape
    x_gradient = np.zeros_like(gray)
    y_gradient = np.zeros_like(gray)

    x_gradient[:, :-1] = np.abs(gray[:, 1:] - gray[:, :-1])
    x_gradient[:, -1] = x_gradient[:, -2]

    y_gradient[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])
    y_gradient[-1, :] = y_gradient[-2, :]

    energy = x_gradient + y_gradient
    return energy


def find_vertical_seam(energy):
    """
    Find the optimal vertical seam DP
    Args:
        energy: array of the energy map
    Returns:
        seam: List of (row, col) tuples representing the seam
    """
    height, width = energy.shape
    M = np.zeros_like(energy)
    M[0, :] = energy[0, :]

    # Cumulative minimum energy
    for i in range(1, height):
        for j in range(width):
            left = M[i - 1, j - 1] if j > 0 else float("inf")
            center = M[i - 1, j]
            right = M[i - 1, j + 1] if j < width - 1 else float("inf")
            M[i, j] = energy[i, j] + min(left, center, right)

    # Backtrack to find the seam
    seam = []
    j = np.argmin(M[-1, :])
    seam.append((height - 1, j))

    for i in range(height - 2, -1, -1):
        left = M[i, j - 1] if j > 0 else float("inf")
        center = M[i, j]
        right = M[i, j + 1] if j < width - 1 else float("inf")
        min_val = min(left, center, right)
        if j > 0 and left == min_val:
            j -= 1
        elif j < width - 1 and right == min_val:
            j += 1
        seam.append((i, j))

    seam.reverse()
    return seam


def find_horizontal_seam(energy):
    """
    Find the optimal horizontal seam by transposing energy map
    Args:
        energy: array of the energy map
    Returns:
        seam: List of (row, col) tuples representing the seam
    """
    seam = find_vertical_seam(energy.T)
    return seam


def remove_vertical_seam(image, seam):
    """
    Remove a vertical seam from the image
    Args:
        image: array of the input image
        seam: List of (row, col) tuples representing the seam
    Returns:
        new_image: array of the image with the seam removed.
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    new_shape = (
        (height, width - 1, channels) if image.ndim == 3 else (height, width - 1)
    )
    new_image = np.zeros(new_shape, dtype=image.dtype)

    for i, j in seam:
        if image.ndim == 3:
            new_image[i, :j, :] = image[i, :j, :]
            new_image[i, j:, :] = image[i, j + 1 :, :]
        else:
            new_image[i, :j] = image[i, :j]
            new_image[i, j:] = image[i, j + 1 :]

    return new_image


def remove_horizontal_seam(image, seam):
    """
    Remove a horizontal seam from the image
    Args:
        image: array of the input image
        seam: List of (row, col) tuples representing the seam.
    Returns:
        new_image: array of the image with the seam removed
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    new_shape = (
        (height - 1, width, channels) if image.ndim == 3 else (height - 1, width)
    )
    new_image = np.zeros(new_shape, dtype=image.dtype)

    for j, i in seam:
        if image.ndim == 3:
            new_image[:i, j, :] = image[:i, j, :]
            new_image[i:, j, :] = image[i + 1 :, j, :]
        else:
            new_image[:i, j] = image[:i, j]
            new_image[i:, j] = image[i + 1 :, j]

    return new_image


def find_multiple_seams(image, num_seams, direction):
    """
    Find multiple seams on the original image for visualization
    Args:
        image: array of the input image.
        num_seams: Number of seams to find
        direction: 'vertical' or 'horizontal'
    Returns:
        seams: List of seams, each a list of (row, col) tuples
    """
    if num_seams <= 0:
        return []

    seams = []
    energy = compute_energy(image)

    for _ in range(num_seams):
        if direction == "vertical":
            seam = find_vertical_seam(energy)
        elif direction == "horizontal":
            seam = find_horizontal_seam(energy)
        else:
            raise ValueError("Direction must be 'vertical' or 'horizontal'")

        seams.append(seam)
        for i, j in seam:
            energy[i, j] = float("inf")

    return seams


def create_seam_visualization(image, vertical_seams, horizontal_seams):
    """
    Create an image visualizing the seams to be removed
    Args:
        image: array of the original image
        vertical_seams: List of vertical seams
        horizontal_seams: List of horizontal seams
    Returns:
        vis_image: array of the image with seams marked
    """
    vis_image = (
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )

    # Mark vertical seams
    for seam in vertical_seams:
        for i, j in seam:
            vis_image[i, j] = [0, 0, 255]

    # Mark horizontal seams
    for seam in horizontal_seams:
        for j, i in seam:
            vis_image[i, j] = [0, 0, 255]

    return vis_image


def seam_carve(image, mode, target_width=None, target_height=None):
    """
    Perform seam carving to resize the image based on chosen mode
    Args:
        image: array of the input image
        mode: String ('horizontal', 'vertical', or 'both)
        target_width: Desired width after resizing
        target_height: Desired height after resizing
    Returns:
        resized_image: array of the resized imag
        vis_image: array of the visualization of removed seams
    """
    height, width = image.shape[:2]

    if mode == "horizontal":
        if target_width is None or target_width >= width:
            raise AssertionError(
                "target_width must be specified and less than original width"
            )
        num_vertical_seams = width - target_width
        num_horizontal_seams = 0
    elif mode == "vertical":
        if target_height is None or target_height >= height:
            raise AssertionError(
                "target_height must be specified and less than original height"
            )
        num_vertical_seams = 0
        num_horizontal_seams = height - target_height
    elif mode == "both":
        if (
            target_width is None
            or target_height is None
            or target_width >= width
            or target_height >= height
        ):
            raise AssertionError(
                "Both target_width and target_height must be specified and less than original dimensions"
            )
        num_vertical_seams = width - target_width
        num_horizontal_seams = height - target_height
    else:
        raise ValueError("Mode must be 'horizontal', 'vertical', or 'both'")

    # Ensure image size constraints are met
    if width > 800 or height > 800:
        raise ValueError("Image dimensions must not exceed 800x800")
    if num_vertical_seams > width // 2 or num_horizontal_seams > height // 2:
        print("Warning: Reduction exceeds half the original size")

    vertical_seams = find_multiple_seams(image, num_vertical_seams, "vertical")
    horizontal_seams = find_multiple_seams(image, num_horizontal_seams, "horizontal")
    vis_image = create_seam_visualization(image, vertical_seams, horizontal_seams)

    resized_image = image.copy()

    for _ in range(num_vertical_seams):
        energy = compute_energy(resized_image)
        seam = find_vertical_seam(energy)
        resized_image = remove_vertical_seam(resized_image, seam)

    for _ in range(num_horizontal_seams):
        energy = compute_energy(resized_image)
        seam = find_horizontal_seam(energy)
        resized_image = remove_horizontal_seam(resized_image, seam)

    return resized_image, vis_image


if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    input_image_path = os.path.join(input_dir, "input_2.jpg")
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    #     modes = [("horizontal", 600, None), ("vertical", None, 600), ("both", 400, 400)]
    modes = [("both", 400, 400)]

    for mode, tw, th in modes:
        start = time.time()

        mode_output_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_output_dir, exist_ok=True)

        resized, vis = seam_carve(input_image, mode, target_width=tw, target_height=th)

        resized_path = os.path.join(mode_output_dir, f"resized_{mode}.jpg")
        vis_path = os.path.join(mode_output_dir, f"vis_{mode}.jpg")

        cv2.imwrite(resized_path, resized)
        cv2.imwrite(vis_path, vis)

        end = time.time()
        print(
            f"Completed {mode} resizing: {resized.shape}, took {end - start:.2f} seconds, saved to {mode_output_dir}"
        )
