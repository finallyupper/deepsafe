import cv2
import numpy as np 
import torch
from umeyama import umeyama

def random_transform(image, 
                     rotation_range, zoom_range, shift_range, random_flip,
                     #random_brightness, random_contrast #Add(Yoojin)
                     ):
    #Added(Yoojin)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]

#    brightness_factor = np.random.uniform(1- random_brightness, 1+random_brightness)
#    contrast_factor = np.random.uniform(1-random_contrast, 1+random_contrast)

#    bright_img = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
#    mean = np.mean(bright_img)
#    result = cv2.addWeighted(bright_img, contrast_factor, bright_img, 0, mean * (1 - contrast_factor))
    return result

def random_warp(image):
    assert image.shape == (256, 256, 3)  # Ensure input is 256x256

    # Generate random displacement maps
    range_ = np.linspace(128 - 80, 128 + 80, 5)  # Grid points
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    # Add random noise to displacement maps
    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)

    # Interpolate the displacement maps to the desired resolution (160x160)
    interp_mapx = cv2.resize(mapx, (160, 160)).astype('float32')  # Changed size to 160x160
    interp_mapy = cv2.resize(mapy, (160, 160)).astype('float32')  # Changed size to 160x160

    # Apply remapping to generate the warped image
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    # Define source and destination points for affine transformation
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:129:32, 0:129:32].T.reshape(-1, 2)  # Adjusted for 160x160

    # Compute affine transformation matrix
    mat = umeyama(src_points, dst_points, True)[0:2]

    # Apply affine transformation to produce the target image
    target_image = cv2.warpAffine(image, mat, (160, 160))  # Resize to 160x160

    return warped_image, target_image
