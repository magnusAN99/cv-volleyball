//
// Created by Magnus on 15-04-2026.
//
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Court {
public:
    static constexpr float WIDTH = 9.0f;
    static constexpr float HEIGHT = 18.0f;
    static constexpr float ATTACK_LINE = 3.0f;

    static std::vector<cv::Point2f> landmarkWorldPositions();
    static std::vector<std::string> landmarkNames();

    //Draw methods
    static cv::Mat draw(int pixelsPerMeter = 80, int margin = 50) ;
    static cv::Mat drawWithPoints(const std::vector<cv::Point2f>& worldPoints, int currentIndex = -1,
        int pixelsPerMeter = 40, int margin = 30) ;
    cv::Mat drawVerification(const cv::Mat& frame) const;
    static cv::Mat drawWithPlayers(const std::vector<cv::Point2f>& worldPoints,
        int pixelsPerMeter = 40, int margin = 30) ;

    const cv::Mat& getHomography() const;

    // Computations
    bool computeHomography(const std::vector<cv::Point2f>& imagePoints,
                           const std::vector<cv::Point2f>& worldPoints);
    cv::Mat warpToTopDown(const cv::Mat& frame,
                          int pixelsPerMeter = 40,
                          int margin = 50) const;
    bool isOnCourt(cv::Point2f pixel, float margin = 1.0f) const;
    cv::Point2f imageToWorld(cv::Point2f pixel) const;
    cv::Point2f worldToImage(cv::Point2f world) const;

private:
    cv::Mat homography_;
};