//
// Created by Magnus on 14-04-2026.
//

#ifndef VOLLEYBALL_CV_VIDEO_H
#define VOLLEYBALL_CV_VIDEO_H

#endif //VOLLEYBALL_CV_VIDEO_H

#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class VideoSource {
    public:
        bool open(const std::string& path);
        cv::Mat getFrame();
        void setFrame(int frameNumber);
        void nextFrame();
        void prevFrame();

        int currentFrameNumber() const;
        int numFrames() const;
        int width() const;
        int height() const;
    private:
        cv::VideoCapture capture_;
        cv::Mat currentFrame_;
        int frameNum_ = 0;
};