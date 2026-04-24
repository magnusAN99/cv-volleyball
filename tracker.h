//
// Created by Magnus on 24-04-2026.
//
#pragma once
#include <opencv2/opencv.hpp>

struct Track {
    int id;
    cv::Point2f worldPosition;
    int framesSinceLastSeen;
};

