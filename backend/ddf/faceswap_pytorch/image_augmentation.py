import cv2
import numpy
import torch
from faceswap_pytorch.umeyama import umeyama

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    #Added(Yoojin)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    h, w = image.shape[0:2]
    rotation = numpy.random.uniform(-rotation_range, rotation_range)
    scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = numpy.random.uniform(-shift_range, shift_range) * w
    ty = numpy.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if numpy.random.random() < random_flip:
        result = result[:, ::-1]
    return result

# def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
#     if isinstance(image, torch.Tensor):
#         image = image.detach().cpu().permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy

#     h, w = image.shape[0:2]
#     rotation = numpy.random.uniform(-rotation_range, rotation_range)
#     scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
#     tx = numpy.random.uniform(-shift_range, shift_range) * w
#     ty = numpy.random.uniform(-shift_range, shift_range) * h
#     mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
#     mat[:, 2] += (tx, ty)
#     result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
#     if numpy.random.random() < random_flip:
#         result = result[:, ::-1]
#     return result


def random_warp(image):
    assert image.shape == (256, 256, 3)  # Ensure input is 256x256

    # Generate random displacement maps
    range_ = numpy.linspace(128 - 80, 128 + 80, 5)  # Grid points
    mapx = numpy.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    # Add random noise to displacement maps
    mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
    mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

    # Interpolate the displacement maps to the desired resolution (160x160)
    interp_mapx = cv2.resize(mapx, (160, 160)).astype('float32')  # Changed size to 160x160
    interp_mapy = cv2.resize(mapy, (160, 160)).astype('float32')  # Changed size to 160x160

    # Apply remapping to generate the warped image
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    # Define source and destination points for affine transformation
    src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = numpy.mgrid[0:129:32, 0:129:32].T.reshape(-1, 2)  # Adjusted for 160x160

    # Compute affine transformation matrix
    mat = umeyama(src_points, dst_points, True)[0:2]

    # Apply affine transformation to produce the target image
    target_image = cv2.warpAffine(image, mat, (160, 160))  # Resize to 160x160

    return warped_image, target_image


# # get pair of random warped images from aligened face image
# def random_warp(image):
#     assert image.shape == (256, 256, 3)
#     range_ = numpy.linspace(128 - 80, 128 + 80, 5)
#     mapx = numpy.broadcast_to(range_, (5, 5))
#     mapy = mapx.T

#     mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
#     mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

#     interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
#     interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

#     warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

#     src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
#     dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
#     mat = umeyama(src_points, dst_points, True)[0:2]

#     target_image = cv2.warpAffine(image, mat, (64, 64))

#     return warped_image, target_image
