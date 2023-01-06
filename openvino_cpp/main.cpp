#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <openvino/openvino.hpp>
#include <openvino/op/region_yolo.hpp>

using namespace std;
// define NMS score
const float NMS_SCORE = 0.1;
// define NMS threshold
const float NMS_THRESHOLD = 0.4;
// define confidence threshold
const float CONFIDENCE_THRESHOLD = 0.4;
const int PADDING_COLOR = 128;
// define image input size
const int INPUT_SIZE = 960;

struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

// custom

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry)
{
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

class DetectionParams
{
    template <typename T>
    void computeAnchors(const std::vector<float> &initialAnchors, const std::vector<T> &mask)
    {
        anchors.resize(num * 2);
        for (int i = 0; i < num; ++i)
        {
            anchors[i * 2] = initialAnchors[mask[i] * 2];
            anchors[i * 2 + 1] = initialAnchors[mask[i] * 2 + 1];
        }
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors;

    DetectionParams() {}

    DetectionParams(const ov::op::v0::RegionYolo &region)
    {
        coords = region.get_num_coords();
        classes = region.get_num_classes();
        const std::vector<float> &initialAnchors = region.get_anchors();
        const std::vector<int64_t> &mask = region.get_mask();
        num = mask.size();

        computeAnchors(initialAnchors, mask);
    }
};

class Detections
{
public:
    template <typename T>
    T &get() const
    {
        return *std::static_pointer_cast<T>(detections);
    }
    template <typename T>
    void set(T *detections)
    {
        this->detections.reset(detections);
    }

private:
    std::shared_ptr<void> detections;
};

struct DetectionObject
{
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) : xmin{static_cast<int>((x - w / 2) * w_scale)},
                                                                                                                            ymin{static_cast<int>((y - h / 2) * h_scale)},
                                                                                                                            xmax{static_cast<int>(this->xmin + w * w_scale)},
                                                                                                                            ymax{static_cast<int>(this->ymin + h * h_scale)},
                                                                                                                            class_id{class_id},
                                                                                                                            confidence{confidence} {}

    bool operator<(const DetectionObject &s2) const
    {
        return this->confidence < s2.confidence;
    }
    bool operator>(const DetectionObject &s2) const
    {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2)
{
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void parseDacovaDetectionOutput(ov::Tensor tensor,
                     const DetectionParams &detectionParams, const unsigned long resized_im_h,
                     const unsigned long resized_im_w, const unsigned long original_im_h,
                     const unsigned long original_im_w,
                     const double threshold, std::vector<DetectionObject> &objects)
{

    const int height = static_cast<int>(tensor.get_shape()[2]);
    const int width = static_cast<int>(tensor.get_shape()[3]);
    if (height != width)
        throw std::runtime_error("Invalid size of output. It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(height) +
                                 ", current W = " + std::to_string(height));

    auto num = detectionParams.num;
    auto coords = detectionParams.coords;
    auto classes = detectionParams.classes;

    auto anchors = detectionParams.anchors;

    auto side = height;
    auto side_square = side * side;
    const float *data = tensor.data<float>();
    
    for (int i = 0; i < side_square; ++i)
    {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n)
        {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = data[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + data[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + data[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(data[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(data[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j)
            {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * data[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                                    static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                                    static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void drawDetections(
    cv::Mat &img,
    const std::vector<DetectionObject> &detections,
    const std::vector<std::string> classes,
    const std::vector<cv::Scalar> &colors)
{
    for (const DetectionObject &f : detections)
    {
        cv::rectangle(img,
                      cv::Rect2f(static_cast<float>(f.xmin),
                                 static_cast<float>(f.ymin),
                                 static_cast<float>((f.xmax - f.xmin)),
                                 static_cast<float>((f.ymax - f.ymin))),
                      colors[static_cast<int>(f.class_id)],
                      2);

        cv::putText(img, classes[f.class_id], cv::Point(f.xmin, f.ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

Resize resize_and_pad(cv::Mat &img, cv::Size new_shape)
{
    float width = img.cols;
    float height = img.rows;
    // cout << "width: " << width << " height: " << height << endl;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    // cout << "new 1 : " << new_unpadW << " new 2" << new_unpadH << endl;
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    // cout << "new dw: " << resize.dw << " new dh: " << resize.dh << endl;
    cv::Scalar color = cv::Scalar(PADDING_COLOR, PADDING_COLOR, PADDING_COLOR);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
}

int main()
{

    std::string model_path = "../../samples/weights/onnx/dacovadetection_20221129_13345_265666.xml";
    std::string img_path = "../../samples/imgs/test3.jpg";
    std::string out_img_path = "../../output/onnx/test3_onnx_cpp.jpg";
    std::vector<std::string> classes = {
        "__ignore__",
        "large",
        "middle",
        "small"};

    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->get_parameters().size() != 1)
    {
        throw std::logic_error("DacovaDetection model must have only one input");
    }
    // cout << "Number of input: " << model->get_parameters().size() << endl;
    // cout << "----------------------------" << endl;
    model->reshape({1, INPUT_SIZE, INPUT_SIZE, 3}); // change input from ?*960*960*3 to 1*960*960*3

    // Step 3. Read input image
    cv::Mat img = cv::imread(img_path);
    // resize image
    Resize res = resize_and_pad(img, cv::Size(INPUT_SIZE, INPUT_SIZE));
    // cout << "------------------" << endl;
    // cout << res.dw << " " << res.dh << endl;

    // Step 4. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp(model);

    // Specify input image format
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255., 255., 255.});

    //  Specify model's input layout
    ppp.input()
        .model()
        .set_layout("NHWC");
    // Specify output results format: multi outputs
    for (const ov::Output<ov::Node> &out : model->outputs())
    {
        ppp.output(out.get_any_name())
            .tensor()
            .set_element_type(ov::element::f32);
    }

    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    // Step 5. Create tensor from image
    float *input_data = (float *)res.resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);

    std::vector<std::pair<ov::Output<ov::Node>, DetectionParams>> detectionParams;
    for (const ov::Output<ov::Node> &out : model->outputs())
    {
        const ov::op::v0::RegionYolo *region = dynamic_cast<ov::op::v0::RegionYolo *>(out.get_node()->get_input_node_ptr(0));
        if (!region)
        {
            throw std::runtime_error("Invalid output type: " + std::string(region->get_type_info().name) + ". Region expected");
        }
        detectionParams.emplace_back(out, *region);
    }

    // Step 6. Create an infer request for model inference
    ov::InferRequest req = compiled_model.create_infer_request();

    using timer = std::chrono::high_resolution_clock;
    timer::time_point lastTime = timer::now();
    req.set_input_tensor(input_tensor);
    req.start_async();
    req.wait();

    // Step 7: Postprocess output
    auto resized_im_h = INPUT_SIZE;
    auto resized_im_w = INPUT_SIZE;
    float ori_width = img.cols;
    float ori_height = img.rows;
    // image size of project
    cv::Size frameSize(INPUT_SIZE, INPUT_SIZE);
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (const std::pair<ov::Output<ov::Node>, DetectionParams> &idxParams : detectionParams)
    {
        parseDacovaDetectionOutput(req.get_tensor(idxParams.first), idxParams.second, resized_im_h, resized_im_w, ori_width, ori_height, NMS_SCORE, objects);
    }
    // Filtering overlapping boxes and lower confidence object
    std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
    cout << "Num object: " << objects.size() <<endl;
    for (size_t i = 0; i < objects.size(); ++i)
    {

        if (objects[i].confidence == 0)
            continue;
        for (size_t j = i + 1; j < objects.size(); ++j)
            if (IntersectionOverUnion(objects[i], objects[j]) >= NMS_THRESHOLD)
                objects[j].confidence = 0;
    }

    std::vector<DetectionObject> output;

    for (auto &object : objects)
    {
        if (object.confidence < CONFIDENCE_THRESHOLD)
            continue;
        output.push_back(object);
    }
    auto currTime = timer::now();
    auto timeInfer = (currTime - lastTime);
    cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeInfer).count() << "ms" << endl;

    // Step 8: Print + visualize
    std::vector<cv::Scalar> colors;
    if (detectionParams.size() > 0)
        for (int i = 0; i < static_cast<int>(detectionParams.front().second.classes); ++i)
            colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    
        
    drawDetections(img, output, classes, colors);
    cv::imwrite(out_img_path, img);

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];

        cout << "Bbox" << i + 1 << ": Class: " << detection.class_id << " "
             << "Confidence: " << detection.confidence << " Scaled coords: [ "
             << "xmin: " << detection.xmin << ", "
             << "ymin: " << detection.ymin << ", "
             << "xmax: " << detection.xmax << ", "
             << "ymax: " << detection.ymax << " ]" << endl;
        
    }

    return 0;
}