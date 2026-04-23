//
// Created by Magnus on 18-04-2026.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

struct Detection {
    cv::Rect box;
    float confidence;
};

class Detector {
    public:
    Detector(const std::string& modelPath);

    std::vector<Detection> detect(const cv::Mat& frame);

    private:
    std::vector<float> preprocess(const cv::Mat& frame,
                                    float& scale, int& padX, int& padY);
    std::vector<Ort::Value> runInference(std::vector<float>& inputTensor);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputs,
                                        int origWidth, int origHeight,
                                        float scale, int padX, int padY);
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;

    int inputWidth_ = 640;
    int inputHeight_ = 640;
};
