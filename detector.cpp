//
// Created by Magnus on 18-04-2026.
//

#include "detector.h"
#include <opencv2/dnn.hpp>

Detector::Detector(const std::string& modelPath) : env_(ORT_LOGGING_LEVEL_WARNING, "volleyball_cv") {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(4);

#ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wModelPath.c_str(), sessionOptions);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);
#endif
    // Query model for its input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputs = session_->GetInputCount();
    for (size_t i = 0; i < numInputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        inputNames_.push_back(name.get());
    }

    size_t numOutputs = session_->GetOutputCount();
    for (size_t i = 0; i < numOutputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.push_back(name.get());
    }

    std::cout << "Model inputs: ";
    for (const auto& n : inputNames_) std::cout << n << std::endl;
    std::cout << "Model outputs: ";
    for (const auto& n : outputNames_) std::cout << n << std::endl;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    float scale;
    int padX, padY;

    std::vector<float> inputTensor = preprocess(frame, scale, padX, padY);
    std::vector<Ort::Value> outputs = runInference(inputTensor);

    return postprocess(outputs, frame.cols, frame.rows, scale, padX, padY);
};

std::vector<float> Detector::preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY) {
    scale = std::min(
        inputWidth_ / static_cast<float>(frame.cols),
        inputHeight_ / static_cast<float>(frame.rows));

    int newWidth = static_cast<int>(scale * frame.cols);
    int newHeight = static_cast<int>(scale * frame.rows);

    padX = (inputWidth_ - newWidth) / 2;
    padY = (inputHeight_ - newHeight) / 2;

    cv::Mat resized;
    cv::Mat letterboxed;
    cv::resize(frame, resized, cv::Size(newWidth, newHeight));
    cv::copyMakeBorder(resized, letterboxed,
        padY, padY, padX, padX,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed,
        1.0/255.0,
        cv::Size(),
        cv::Scalar(),
        true,
        false
        );

    std::vector<float> tensor(blob.total());
    std::memcpy(tensor.data(), blob.ptr<float>(), blob.total() * sizeof(float));
    return tensor;
}

std::vector<Ort::Value> Detector::runInference(std::vector<float>& inputTensor) {
    std::array<int64_t, 4> inputShape = {1, 3, inputHeight_, inputWidth_};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensor.data(),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size());

    std::vector<const char*> inputNamesCStr;
    for (const auto& n : inputNames_) inputNamesCStr.push_back(n.c_str());

    std::vector<const char*> outputNamesCStr;
    for (const auto& n : outputNames_) outputNamesCStr.push_back(n.c_str());

    return session_->Run(
        Ort::RunOptions{nullptr},
        inputNamesCStr.data(), &input, 1,
        outputNamesCStr.data(), outputNamesCStr.size());
}

std::vector<Detection> Detector::postprocess(std::vector<Ort::Value> &outputs,
    int origWidth, int origHeight,
    float scale, int padX, int padY) {

    // Constants
    const float confThreshold = 0.25f;
    const float nmsThreshold = 0.45f;
    const int personClass = 0;

    auto* data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    int numDetections = static_cast<int>(shape[2]);
    int rowStride = numDetections;

    // vectors for storing results
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int i = 0; i < numDetections; i++) {
        float conf = data[(4 + personClass) * rowStride + i];
        if (conf < confThreshold) continue;

        float cx = data[0 * rowStride + i] * inputWidth_;
        float cy = data[1 * rowStride + i] * inputHeight_;
        float w = data[2 * rowStride + i] * inputWidth_;
        float h = data[3 * rowStride + i] * inputHeight_;

        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        x = (x - padX) / scale;
        y = (y - padY) / scale;
        w = w / scale;
        h = h / scale;

        boxes.push_back(cv::Rect(
            static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(w), static_cast<int>(h)));

        confidences.push_back(conf);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx]});
    }

    return detections;
}
