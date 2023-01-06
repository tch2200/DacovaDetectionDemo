from openvino.runtime import Core
import numpy as np
import os, json, time
from pathlib import Path
from PIL import Image
from common.postprocess_np import detection_postprocess_np
from common.data_utils import image_preprocessing
from common.utils import get_colors, draw_boxes

class INFER(object):
    def __init__(self, path_xml):
        self.path_xml = path_xml
        self.model_input_shape = None
        self.class_names = None
        self.anchors_ = [
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
        self.colors = None
        self.score = 0.5
        self.iou = 0.4
        self.elim_grid_sense = False
        self._generate_model()

    def _generate_model(self):
        ie = Core()
        if not Path(self.path_xml).is_file():
            w = next(Path(self.path_xml).glob("*.xml"))
        network = ie.read_model(
            model=self.path_xml, weights=Path(self.path_xml).with_suffix(".bin")
        )
        self.inference_model = ie.compile_model(model=network, device_name="CPU")

        config_file = Path(self.path_xml).with_suffix(".json")

        with open(config_file, "r", encoding="utf8") as f:
            config = json.load(f)
            self.score = config["model_config"]["probability"]
            self.iou = config["model_config"]["iou"]
            self.class_names = config["model_config"]["class_name"]
            self.model_input_shape = (
                config["model_config"]["image_size"][0],
                config["model_config"]["image_size"][0],
            )

        self.colors = get_colors(len(self.class_names))

    def detect_image(self, image):
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_input_shape[1] % 32 == 0, "Multiples of 32 required"

        image_data = image_preprocessing(image, self.model_input_shape)        
        image_shape = image.size[::-1]

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)

        print("Found {} boxes for {}".format(len(out_boxes), "img"))        
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        # draw result on input image
        image_array = np.array(image, dtype="uint8")
        image_array = draw_boxes(
            image_array,
            out_boxes,
            out_classes,
            out_scores,
            self.class_names,
            self.colors,
        )

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

    def anchors(self, anchors):
        return np.array(anchors).reshape(-1, 2)

    def predict(self, image_data, image_shape):
        pred = self.inference_model([image_data])

        preds = []
        for k in pred.keys():
            preds.append(pred[k])
        out_boxes, out_classes, out_scores = detection_postprocess_np(
            preds,
            image_shape,
            self.anchors(self.anchors_),
            len(self.class_names),
            self.model_input_shape,
            max_boxes=50,
            confidence=self.score,
            iou_threshold=self.iou,
            elim_grid_sense=self.elim_grid_sense,
        )

        return out_boxes, out_classes, out_scores

if __name__ == "__main__":
    # Read model and model config
    detect = INFER(
        # medium model
        path_xml="./samples/weights/openvino/dacovadetection_20221129_13345_265666.xml"        
    )
    while True:
        img = input("Image filename:")
        try:
            image = Image.open(img).convert("RGB")
        except:
            print("file not found!")
            continue
        else:
            base_name = os.path.basename(img)
            r_image, _, _, _ = detect.detect_image(image)
            r_image.show()
            path_save = os.path.join("output", "python", base_name.split(".")[0] + "_python.jpg")            
            r_image.save(path_save)
