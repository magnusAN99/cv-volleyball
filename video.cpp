//
// Created by Magnus on 14-04-2026.
//
#include "video.h"

bool VideoSource::open(const std::string& path) {
    if (!capture_.open(path)) return false;

    capture_ >> currentFrame_;
    frameNum_ = 0;
    return !currentFrame_.empty();
}
cv::Mat VideoSource::getFrame() {
    return currentFrame_;
}

void VideoSource::setFrame(int frameNumber) {
    capture_.set(cv::CAP_PROP_POS_FRAMES, frameNumber);
    capture_ >> currentFrame_;
    frameNum_ = frameNumber;
}

void VideoSource::nextFrame() {
    capture_ >> currentFrame_;
    if (!currentFrame_.empty()) frameNum_++;
}

void VideoSource::prevFrame() {
    if (frameNum_ > 0) setFrame(frameNum_ - 1);
}

int VideoSource::currentFrameNumber() const { return frameNum_; }

int VideoSource::numFrames() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
}

int VideoSource::width() const { return currentFrame_.cols; }
int VideoSource::height() const { return currentFrame_.rows; }