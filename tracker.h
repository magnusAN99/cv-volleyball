//
// Created by Magnus on 24-04-2026.
//
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Track {
    int id;
    cv::Point2f worldPosition;
    int framesSinceLastSeen;
};

class Tracker {
public:
    void update(const std::vector<cv::Point2f>& detections);
    std::vector<cv::Point2f> getActivePositions(int minAge = 5) const;

private:
    std::vector<Track> tracks_;
    int nextId_ = 0;

    float maxMatchDistance_ = 2.0f;
    float alpha_ = 0.3f;
    int maxStaleFrames_ = 30;
};