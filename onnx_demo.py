import onnxruntime as rt
import numpy as np
import os, json, time
from pathlib import Path
from common.postprocess_np import detection_postprocess_np_onnx
from common.data_utils import preprocess_image_onnx
from common.utils import get_colors, draw_boxes
from typing import Tuple, List
import cv2


class INFER(object):
    def __init__(self, onnx_file, device):
        self.onnx_file = onnx_file
        self.device = device
        self.model_input_shape = None
        self.class_names = None
        anchors_ = [
            10.0,
            13.0,
            16.0,
            30.0,
            33.0,
            23.0,
            30.0,
            61.0,
            62.0,
            45.0,
            59.0,
            119.0,
            116.0,
            90.0,
            156.0,
            198.0,
            373.0,
            326.0,
        ]
        self.anchors = np.array(anchors_).reshape(-1, 2)
        self.colors = None
        self.score = 0.8
        self.iou = 0.4
        self.elim_grid_sense = False
                
        self.model_input_shape = None
        self.class_names = None     
        self.model = self._generate_model()
        self.input_name = self.model.get_inputs()[0].name

    def _generate_model(self):
        assert self.device in ["cpu", "gpu"], "{} not in allowed device list".format(
            self.device
        )
        sess_options = rt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = rt.InferenceSession(
            self.onnx_file,
            sess_options=sess_options,
            providers=[
                "CPUExecutionProvider"
                if self.device == "cpu"
                else "CUDAExecutionProvider"
            ],
        )
        config_file = Path(self.onnx_file).with_suffix(".json")

        with open(config_file, "r", encoding="utf8") as f:
            config = json.load(f)
            self.score = config["model_config"]["probability"]
            self.iou = config["model_config"]["iou"]
            self.class_names = config["model_config"]["class_name"]
            self.model_input_shape = [
                config["model_config"]["image_size"][0],
                config["model_config"]["image_size"][0],
            ]
        print("self.score:", self.score)
        print("self.class_names:", self.class_names)
        self.colors = get_colors(len(self.class_names))
        return session

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, List]:
        """Preprocess image before inference
        Args:
            img (np.ndarray): BGR cv2 image
        Returns:
            np.ndarray: image ready to be inferenced
        """
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_input_shape[1] % 32 == 0, "Multiples of 32 required"

        origin_shape = img.shape[:2]
        x = preprocess_image_onnx(image=img, model_input_shape=self.model_input_shape)
        return x, origin_shape

    def _postprocess(self, preds: List[np.ndarray], origin_shape: List) -> List:

        out_boxes, out_classes, out_scores = detection_postprocess_np_onnx(
            preds,
            origin_shape,
            self.anchors,
            len(self.class_names),
            self.model_input_shape,
            max_boxes=50,
            confidence=self.score,
            iou_threshold=self.iou,
            elim_grid_sense=self.elim_grid_sense,
        )

        return out_boxes, out_classes, out_scores

    def __call__(self, img: np.ndarray):
        """Inference imgage for bound boxes.
        Args
            img (np.ndarray): RGB opencv image
            img_size (int, optional): image size when inference. Defaults to 640.
            draw (bool, optional): return img with detected bbox
        Returns:
            np.ndarray: [0] is 'xmin', 'ymin', 'xmax', 'ymax','confidence', 'classid'
        """

        x, origin_shape = self._preprocess(img)

        # input model: RGB mode
        preds = self.model.run(None, {self.input_name: x})

        out_boxes, out_classes, out_scores = self._postprocess(preds, origin_shape)
        return out_boxes, out_classes, out_scores


if __name__ == "__main__":
    # Read model and model config
    detect = INFER(onnx_file="./samples/weights/onnx/dacovadetection_20221212_114558_84403.onnx", device="cpu")

    img_path = "./samples/imgs/test2.jpg"
    if not os.path.exists(img_path):
        print("Img path not found")
    else:

        image = cv2.imread(img_path)  # BGR

        base_name = os.path.basename(img_path)
        t1 = time.time()
        out_boxes, out_classes, out_scores = detect(image)
        print("Time: ", time.time() - t1)        
        path_save = os.path.join("output", "python", base_name.split(".")[0] + "_python.jpg")
        image_array = np.array(image, dtype="uint8")
        r_image = draw_boxes(
            image_array,
            out_boxes,
            out_classes,
            out_scores,
            detect.class_names,
            detect.colors,
        )        
        cv2.imwrite(path_save, r_image)
        print("Save img output at: ", path_save)
