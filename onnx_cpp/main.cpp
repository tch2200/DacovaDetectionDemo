#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>
#include <cassert>

using namespace std;
// define NMS score
const float NMS_SCORE = 0.1;
// define NMS threshold
const float NMS_THRESHOLD = 0.4;
// define confidence threshold
const float CONFIDENCE_THRESHOLD = 0.4;
const int PADDING_COLOR = 114;
// define image input size
const int INPUT_SIZE = 960;

typedef std::pair<float *, std::vector<int64_t>> DataOutputType;

std::string toString(const ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    {
        return "float";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    {
        return "uint8_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    {
        return "int8_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    {
        return "uint16_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    {
        return "int16_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    {
        return "int32_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    {
        return "int64_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    {
        return "string";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    {
        return "bool";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    {
        return "float16";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    {
        return "double";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    {
        return "uint32_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    {
        return "uint64_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    {
        return "complex with float32 real and imaginary components";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    {
        return "complex with float64 real and imaginary components";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    {
        return "complex with float64 real and imaginary components";
    }
    default:
        return "undefined";
    }
}

struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

size_t vectorProduct(const std::vector<int64_t> &vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto &element : vector)
        product *= element;

    return product;
}

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
                                                                                                                            class_id{class_id}, confidence{confidence} {}

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

vector<pair<float, float>> parseAnchorPairList(vector<float> anchorList)
{
    // anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  ~ (30*30, 60*60, 120*120 features)
    vector<pair<float, float>> anchorPairList;
    // vector<vector<float, float>> anchorPerLayer;
    for (int i = 0; i < anchorList.size() - 1; i = i + 2)
    {
        auto pairAnchor = make_pair(anchorList[i], anchorList[i + 1]);
        anchorPairList.push_back(pairAnchor);
    }
    return anchorPairList;
}

vector<pair<float, float>> getAnchorSetByStride(vector<pair<float, float>> anchorPairList, const int stride)
{
    vector<pair<float, float>> anchorSet;

    if (stride == 32)
    {
        anchorSet.push_back(anchorPairList[6]);
        anchorSet.push_back(anchorPairList[7]);
        anchorSet.push_back(anchorPairList[8]);
    }
    else if (stride == 16)
    {
        anchorSet.push_back(anchorPairList[3]);
        anchorSet.push_back(anchorPairList[4]);
        anchorSet.push_back(anchorPairList[5]);
    }
    else if (stride == 8)
    {
        anchorSet.push_back(anchorPairList[0]);
        anchorSet.push_back(anchorPairList[1]);
        anchorSet.push_back(anchorPairList[2]);
    }
    else
    {
        exit(-1);
    }
    return anchorSet;
}

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

#define SIGMOID_CUTOFF_BOTTOM (-6.0f)
#define SIGMOID_CUTOFF_TOP (6.0f)
#define SIGMOID_TABLE_RANGE (1000)

static float sigmoid_table[SIGMOID_TABLE_RANGE];

// init sigmoid value table, used between
// [SIGMOID_CUTOFF_BOTTOM, SIGMOID_CUTOFF_TOP]
//
// need to be called at model init stage
static inline void sigmoid_fast_init()
{
    for (int i = 0; i < SIGMOID_TABLE_RANGE; i++)
    {
        sigmoid_table[i] = sigmoid(SIGMOID_CUTOFF_BOTTOM + i * (SIGMOID_CUTOFF_TOP - SIGMOID_CUTOFF_BOTTOM) / SIGMOID_TABLE_RANGE);
    }

    return;
}

static inline float sigmoid_fast(float x)
{
    if (x <= SIGMOID_CUTOFF_BOTTOM)
    {
        return 0;
    }
    else if (x >= SIGMOID_CUTOFF_TOP)
    {
        return 1;
    }
    else
    {
        int index = round((x - SIGMOID_CUTOFF_BOTTOM) * SIGMOID_TABLE_RANGE / (SIGMOID_CUTOFF_TOP - SIGMOID_CUTOFF_BOTTOM));
        return sigmoid_table[index];
    }
}

void dacovaPostprocessFast(
    const DataOutputType output,
    const int input_width,
    const int input_height,
    const int num_classes,
    const std::vector<std::pair<float, float>> anchors,
    std::vector<DetectionObject> &prediction_list,
    float conf_threshold,
    int ori_width,
    int ori_height)
{

    const float *data = output.first;

    auto batch = output.second[0];
    auto channel = output.second[3];
    auto height = output.second[1];
    auto width = output.second[2];

    int stride = input_width / width;
    auto unit = sizeof(float);
    int anchor_num_per_layer = anchors.size();

    // now we only support single image postprocess
    assert(batch == 1);

    // the featuremap channel should be like 3*(num_classes + 5)
    assert(anchor_num_per_layer * (num_classes + 5) == channel);

    int bytesPerRow, bytesPerImage, bytesPerBatch;

    bytesPerRow = channel * unit;
    bytesPerImage = width * bytesPerRow;
    bytesPerBatch = height * bytesPerImage;

    for (int b = 0; b < batch; b++)
    {
        auto bytes = data + b * bytesPerBatch / unit;        

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int anc = 0; anc < anchor_num_per_layer; anc++)
                {
                    // check bbox score and objectness first to filter invalid prediction
                    int bbox_obj_offset, bbox_scores_offset, bbox_scores_step;

                    bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                    bbox_scores_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 5;
                    bbox_scores_step = 1;

                    float bbox_obj = sigmoid_fast(bytes[bbox_obj_offset]);

                    // get anchor output confidence (class_score * objectness) and filter with threshold
                    float max_conf = 0.0;
                    int max_index = -1;
                    for (int i = 0; i < num_classes; i++)
                    {
                        float tmp_conf = 0.0;

                        // check if only 1 class for different score
                        if (num_classes == 1)
                        {
                            tmp_conf = bbox_obj;
                        }
                        else
                        {
                            tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;
                        }

                        if (tmp_conf > max_conf)
                        {
                            max_conf = tmp_conf;
                            max_index = i;
                        }
                    }
                    if (max_conf >= conf_threshold)
                    {
                        // got a valid prediction, decode bbox and form up data to push to result vector
                        int bbox_x_offset, bbox_y_offset, bbox_w_offset, bbox_h_offset;

                        // Tensorflow format tensor, NHWC
                        bbox_x_offset = h * width * channel + w * channel + anc * (num_classes + 5);
                        bbox_y_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 1;
                        bbox_w_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 2;
                        bbox_h_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 3;

                        // Decode bbox predictions
                        float bbox_x, bbox_y;

                        bbox_x = (sigmoid(bytes[bbox_x_offset]) + w) / width;
                        bbox_y = (sigmoid(bytes[bbox_y_offset]) + h) / height;

                        float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / input_width;
                        float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / input_height;

                        DetectionObject bbox_prediction(bbox_x * input_width, bbox_y * input_width, bbox_h * input_width, bbox_w * input_width, max_index, max_conf,
                                                        static_cast<float>(ori_height) / static_cast<float>(input_height),
                                                        static_cast<float>(ori_width) / static_cast<float>(input_width));

                        prediction_list.emplace_back(bbox_prediction);
                    }
                }
            }
        }
    }

    return;
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
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    cv::Scalar color = cv::Scalar(PADDING_COLOR, PADDING_COLOR, PADDING_COLOR);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
}

Resize preprocess(const cv::Mat &img, cv::Size target_size)
{

    // bgr to rgb
    cv::Mat rgbImg;
    cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB); // HWC - RGB - UINT8

    // resize
    Resize resize = resize_and_pad(rgbImg, target_size);

    // normalize
    resize.resized_image.convertTo(resize.resized_image, CV_32FC3, 1.0 / 255); //  /255

    return resize;
}

void saveFile(const string &filename, const cv::Mat &img)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "Result" << img;
    fs.release();
}

int main()
{

    std::string modelFilePath = "../../samples/weights/onnx/dacovadetection_20221212_114558_84403.onnx";
    std::string imgPath = "../../samples/imgs/test2.jpg";
    std::string out_img_path = "../../output/onnx/out_test2.jpg";
    std::vector<std::string> classes = {
        "__ignore__",
        "large",
        "middle",
        "small"};
    int num_classes = classes.size();
    vector<float> anchorList = {
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
    };    
    vector<pair<float, float>> anchorPairList = parseAnchorPairList(anchorList);

    sigmoid_fast_init();

    cv::Size targetSize(INPUT_SIZE, INPUT_SIZE);

    // Step 1. Initialize OpenVINO Runtime core
    const int batchSize = 1;
    string instanceName{"DacovaDetection-onnx"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilePath.c_str(), sessionOptions);

    // get input - output model infos
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    cout << "Num input nodes: " << numInputNodes << endl;
    cout << "Num output nodes: " << numOutputNodes << endl;

    // get output model infor
    const char *inputName = session.GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to "
                  << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }
    vector<const char *> inputNames{inputName};

    vector<vector<int64_t>> outputDimsList;
    vector<char *> outputNames;
    outputNames.reserve(numOutputNodes);
    outputDimsList.reserve(numOutputNodes);

    for (size_t idx = 0; idx < numOutputNodes; idx++)
    {
        char *outputName = session.GetOutputName(idx, allocator);

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(idx);
        ONNXTensorElementDataType outputType = outputTypeInfo.GetTensorTypeAndShapeInfo().GetElementType();
        vector<int64_t> outputDims = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (outputDims.at(0) == -1)
        {
            std::cout << "Got dynamic batch size. Setting output batch size to "
                      << batchSize << "." << std::endl;
            outputDims.at(0) = batchSize;
        }

        outputNames.emplace_back(outputName);
        outputDimsList.emplace_back(outputDims);
    }

    // Read + preprocessing input image
    cv::Mat img = cv::imread(imgPath); // HWC - BGR - UINT8
    Resize resize = preprocess(img, targetSize);
    cv::Mat processedImg = resize.resized_image;    

    vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault);

    size_t inputTensorSize = vectorProduct(inputDims);
    float *inputTensorValues = (float *)processedImg.data;

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues,
        inputTensorSize,
        inputDims.data(),
        inputDims.size()));

    // inference
    using timer = std::chrono::high_resolution_clock;
    timer::time_point lastTime = timer::now();
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputTensors.data(),
        1,
        outputNames.data(),
        numOutputNodes);

    vector<DataOutputType> outputData;
    outputData.reserve(numOutputNodes);
    int count = 1;
    for (auto &elem : outputTensors)
    {
        cout << "type of input " << count++ << ": " << toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str() << endl;
        outputData.emplace_back(
            std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
    }

    // Step 7: Postprocess output
    auto resized_im_h = INPUT_SIZE;
    auto resized_im_w = INPUT_SIZE;
    float ori_width = img.cols;
    float ori_height = img.rows;
    cv::Size frameSize(960, 960);
    std::vector<DetectionObject> objects;
    for (int layer_id = 0; layer_id < numOutputNodes; layer_id++)
    {
        int sizeFeatureMap = outputData[layer_id].second[1];
        int stride = INPUT_SIZE / sizeFeatureMap;
        vector<pair<float, float>> anchorSet = getAnchorSetByStride(anchorPairList, stride);
        dacovaPostprocessFast(
            outputData[layer_id],
            resized_im_w,
            resized_im_h,
            num_classes,
            anchorSet,
            objects,
            NMS_SCORE,
            ori_width,
            ori_height);        
    }

    // Parsing outputs
    cout << "Num objects after parseOutput: " << objects.size() << endl;
    // Filtering overlapping boxes and lower confidence object
    std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());

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
    cout << "Num object after postprocess: " << output.size() << endl;

    auto currTime = timer::now();
    auto timeInfer = (currTime - lastTime);
    cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeInfer).count() << "ms" << endl;

    // Step 8: Print + visualize
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

    std::vector<cv::Scalar> colors;
    for (int i = 0; i < num_classes; ++i)
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));

    drawDetections(img, output, classes, colors);

    cv::imwrite(out_img_path, img);
    cout << "Saved result at: " << out_img_path << endl;

    return 0;
}