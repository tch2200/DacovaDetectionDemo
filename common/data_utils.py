import numpy as np
from PIL import Image
import cv2

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def image_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w / src_w, target_h / src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w) // 2
    dy = (target_h - padding_h) // 2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new("RGB", target_size, (128, 128, 128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image

def image_resize_onnx(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            cv2 object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: cv2 image

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return im
    # return im, ratio, (dw, dh)

def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image.astype(np.float32) / 255.0

    return image

def image_preprocessing(image, model_input_shape):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_input_shape: model input image shape
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image_resize(image, model_input_shape[::-1])
    image_data = np.asarray(resized_image).astype("float32")
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)
    
    return image_data

def preprocess_image_onnx(image, model_input_shape):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            cv2 object containing image data
        model_input_shape: model input image shape
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    # resized_image = letterbox_resize(image, model_input_shape[::-1])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = image_resize_onnx(image_rgb, model_input_shape)
    image_data = np.asarray(resized_image).astype("float32")
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data