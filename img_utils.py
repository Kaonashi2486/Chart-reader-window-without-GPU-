import cv2
import numpy as np
import random

# Convert a color image to grayscale. 
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize the image by subtracting the mean and dividing by the standard deviation.
def normalize_(image, mean, std):
    image -= mean
    image /= std

# Blend two images by a specified ratio.
# alpha: The blending ratio.
# image1, image2: The images to blend.
def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

# Adjust lighting of the image based on random alpha values.
# data_rng: Data range or random generator.
# image: The image to modify.
# alphastd: Standard deviation controlling lighting variation.
# eigval, eigvec: Eigenvalues and eigenvectors used to compute lighting variation.
def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)

# Apply color jittering to the image, including brightness, contrast, and saturation variations.
def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

# Adjust the saturation of the image.
# data_rng: Data range or random generator.
# image: The image to modify.
# gs: Grayscale image.
# gs_mean: Mean of grayscale image.
# var: Variable controlling the saturation change.
def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

# Adjust the brightness of the image.
def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

# Adjust the contrast of the image.
def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

# Crop an image centered around a specified point and of a specified size.
# Returns the cropped image, the border, and the offset.
def crop_image(image, center, size):
    cty, ctx = center
    height, width = size
    im_height, im_width = image.shape[0:2]
    cropped_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    
    # Calculate boundaries of the crop region within the original image.
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)
    
    # Calculate the distances from the center to the crop boundaries.
    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty
    
    # Calculate the center of the cropped image.
    cropped_cty, cropped_ctx = height // 2, width // 2
    
    # Create slices for the cropped region.
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    
    # Apply the crop.
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]
    
    # Create an array representing the boundaries of the cropped region.
    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)
    
    # Calculate the offset of the crop.
    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset
