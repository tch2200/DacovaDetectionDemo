import numpy as np
from enum import Enum
import cv2, colorsys

def get_colors(number, bright=True):
    """
    Generate random colors for drawing bounding boxes.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1.0, brightness) for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
    )
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors

labelType = Enum(
    "labelType",
    (
        "LABEL_TOP_OUTSIDE",
        "LABEL_BOTTOM_OUTSIDE",
        "LABEL_TOP_INSIDE",
        "LABEL_BOTTOM_INSIDE",
    ),
)

def draw_label(image, text, color, coords, label_type=labelType.LABEL_TOP_OUTSIDE):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    if (
        label_type == labelType.LABEL_TOP_OUTSIDE
        or label_type == labelType.LABEL_BOTTOM_INSIDE
    ):
        cv2.rectangle(
            image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED
        )
        cv2.putText(
            image,
            text,
            (x + padding, y - text_height + padding),
            font,
            fontScale=font_scale,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.rectangle(
            image, (x, y), (x + rect_width, y + rect_height), color, cv2.FILLED
        )
        cv2.putText(
            image,
            text,
            (x + padding, y + text_height + padding),
            font,
            fontScale=font_scale,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )

    return image

def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        if show_score:
            label = "{} {:.2f}".format(class_name, score)
        else:
            label = "{}".format(class_name)

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0, 0, 0)
        else:
            color = colors[cls]

        # choose label type according to box size
        if ymin > 20:
            label_coords = (xmin, ymin)
            label_type = label_type = labelType.LABEL_TOP_OUTSIDE
        elif ymin <= 20 and ymax <= image.shape[0] - 20:
            label_coords = (xmin, ymax)
            label_type = label_type = labelType.LABEL_BOTTOM_OUTSIDE
        elif ymax > image.shape[0] - 20:
            label_coords = (xmin, ymin)
            label_type = label_type = labelType.LABEL_TOP_INSIDE

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, label_coords, label_type)

    return image
