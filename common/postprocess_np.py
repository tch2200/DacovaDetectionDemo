import numpy as np

from common.detection_postprocess_np import (
    detection_decode,
    detection_handle_predictions,
    detection_correct_boxes,
    detection_adjust_boxes,
)



def detection_embed_decode(
    predictions, anchors, num_classes, input_shape, elim_grid_sense=False
):
    """decode 3 layer outputs

    Args:
        predictions (_type_): _description_
        anchors (_type_): _description_
        num_classes (_type_): _description_
        input_shape (_type_): _description_
        elim_grid_sense (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert (
        len(predictions) == len(anchors) // 3
    ), "anchor numbers does not match prediction."    

    if len(predictions) == 3:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    elif len(predictions) == 2:
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]
    else:
        raise ValueError("Unsupported prediction length: {}".format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):

        prediction = prediction.transpose((0, 2, 3, 1))
        out = detection_decode(
            prediction,
            anchors[anchor_mask[i]],
            num_classes,
            input_shape,
            scale_x_y=scale_x_y[i],
            use_softmax=False,
        )
        results.append(out)

    return np.concatenate(results, axis=1)

def detection_embed_decode_onnx(
    predictions, anchors, num_classes, input_shape, elim_grid_sense=False
):
    """decode 3 layer outputs

    Args:
        predictions (_type_): _description_
        anchors (_type_): _description_
        num_classes (_type_): _description_
        input_shape (_type_): _description_
        elim_grid_sense (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert (
        len(predictions) == len(anchors) // 3
    ), "anchor numbers does not match prediction."    

    if len(predictions) == 3:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    elif len(predictions) == 2:
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]
    else:
        raise ValueError("Unsupported prediction length: {}".format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        
        out = detection_decode(
            prediction,
            anchors[anchor_mask[i]],
            num_classes,
            input_shape,
            scale_x_y=scale_x_y[i],
            use_softmax=False,
        )
        results.append(out)

    return np.concatenate(results, axis=1)

def detection_postprocess_np(
    detection_outputs,
    image_shape,
    anchors,
    num_classes,
    model_input_shape,
    max_boxes=50,
    confidence=0.2,
    iou_threshold=0.5,
    elim_grid_sense=False,
):
    """_summary_

    Args:
        detection_outputs ([list[np.ndarray]]): list 3 matrix output
        image_shape (_type_): _description_
        anchors (_type_): _description_
        num_classes (_type_): _description_
        model_input_shape (_type_): _description_
        max_boxes (int, optional): _description_. Defaults to 50.
        confidence (float, optional): _description_. Defaults to 0.2.
        iou_threshold (float, optional): _description_. Defaults to 0.5.
        elim_grid_sense (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    detection_outputs.sort(key=lambda x: x.shape[1])

    predictions = detection_embed_decode(
        detection_outputs,
        anchors,
        num_classes,
        input_shape=model_input_shape,
        elim_grid_sense=elim_grid_sense,
    )

    predictions = detection_correct_boxes(predictions, image_shape, model_input_shape)

    boxes, classes, scores = detection_handle_predictions(
        predictions,
        image_shape,
        num_classes,
        max_boxes=max_boxes,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )

    boxes = detection_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

def detection_postprocess_np_onnx(
    detection_outputs,
    image_shape,
    anchors,
    num_classes,
    model_input_shape,
    max_boxes=50,
    confidence=0.2,
    iou_threshold=0.5,
    elim_grid_sense=False,
):
    """_summary_

    Args:
        detection_outputs ([list[np.ndarray]]): list 3 matrix output
        image_shape (_type_): _description_
        anchors (_type_): _description_
        num_classes (_type_): _description_
        model_input_shape (_type_): _description_
        max_boxes (int, optional): _description_. Defaults to 50.
        confidence (float, optional): _description_. Defaults to 0.2.
        iou_threshold (float, optional): _description_. Defaults to 0.5.
        elim_grid_sense (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    detection_outputs.sort(key=lambda x: x.shape[1])

    predictions = detection_embed_decode_onnx(
        detection_outputs,
        anchors,
        num_classes,
        input_shape=model_input_shape,
        elim_grid_sense=elim_grid_sense,
    )

    predictions = detection_correct_boxes(predictions, image_shape, model_input_shape)

    boxes, classes, scores = detection_handle_predictions(
        predictions,
        image_shape,
        num_classes,
        max_boxes=max_boxes,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )

    boxes = detection_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores
