//
// Created by Magnus on 15-04-2026.
//

#include "court.h"

cv::Mat Court::draw(const int pixelsPerMeter, const int margin) {
    const int courtW = static_cast<int>(WIDTH * pixelsPerMeter);
    const int courtH = static_cast<int>(HEIGHT * pixelsPerMeter);
    const int attackLineOffset = static_cast<int>(ATTACK_LINE * pixelsPerMeter);
    const int centerLineOffset = static_cast<int>((HEIGHT / 2) * pixelsPerMeter);

    cv::Mat court(courtH + 2*margin, courtW + 2*margin,
                 CV_8UC3, cv::Scalar(40,40,40));

    cv::rectangle(court,
        cv::Point(margin, margin),
        cv::Point(margin + courtW, margin + courtH),
        cv::Scalar(255, 255, 255), 2);

    cv::line(court,
        cv::Point(margin, margin + centerLineOffset - attackLineOffset),
        cv::Point(margin + courtW, margin + centerLineOffset - attackLineOffset),
        cv::Scalar(255, 255, 255), 2);

    cv::line(court,
        cv::Point(margin, margin  + centerLineOffset),
        cv::Point(margin + courtW, margin + centerLineOffset),
        cv::Scalar(255, 255, 255), 1);

    cv::line(court,
        cv::Point(margin, margin + centerLineOffset + attackLineOffset),
        cv::Point(margin + courtW, margin + centerLineOffset + attackLineOffset),
        cv::Scalar(255, 255, 255), 1);

    return court;
}

bool Court::computeHomography(const std::vector<cv::Point2f>& imagePoints,
    const std::vector<cv::Point2f>& worldPoints) {
    if (imagePoints.size() != worldPoints.size() || imagePoints.size() < 4) {
        return false;
    }
    homography_ = cv::findHomography(imagePoints, worldPoints, cv::RANSAC);
    return !homography_.empty();
}

cv::Point2f Court::imageToWorld(const cv::Point2f pixel) const {
    const std::vector<cv::Point2f> in = { pixel };
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(in, out, homography_);
    return out[0];
}

cv::Point2f Court::worldToImage(const cv::Point2f world) const {
    const std::vector<cv::Point2f> in = { world };
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(in, out, homography_.inv());
    return out[0];
}

std::vector<cv::Point2f> Court::landmarkWorldPositions() {
    return
    {cv::Point2f(0, 0), cv::Point2f(9, 0),
    cv::Point2f(0, 6), cv::Point2f(9, 6),
    cv::Point2f(0, 9), cv::Point2f(9, 9),
    cv::Point2f(0, 12), cv::Point2f(9, 12),
    cv::Point2f(0, 18), cv::Point2f(9, 18)};
};

std::vector<std::string> Court::landmarkNames() {
    return
    {"nearBaselineLeft", "nearBaselineRight",
    "nearAttackLeft", "nearAttackRight",
    "centerLeft", "centerRight",
    "farAttackLeft", "farAttackRight",
    "farBaselineRight", "farBaselineLeft"};
};

cv::Mat Court::drawWithPoints(const std::vector<cv::Point2f>& worldPoints, int currentIndex,
        int pixelsPerMeter, int margin) {

    cv::Mat court = draw(pixelsPerMeter, margin);
    const int courtH = static_cast<int>(HEIGHT * pixelsPerMeter);
    for (int i = 0; i < worldPoints.size(); i++) {
        int px = margin + static_cast<int>(worldPoints[i].x * pixelsPerMeter);
        int py = margin + courtH - static_cast<int>(worldPoints[i].y * pixelsPerMeter);
        cv::Point center(px, py);

        if (i < currentIndex) {
            cv::circle(court, center, 8, cv::Scalar(0, 255, 0), -1);
        }
        else if (i == currentIndex) {
            cv::circle(court, center, 8, cv::Scalar(255, 0, 0), -1);
        }
        else {
            cv::circle(court, center, 8, cv::Scalar(0, 0, 0), -1);
        }
    }

    return court;
}

cv::Mat Court::drawWithPlayers(const std::vector<cv::Point2f> &worldPoints, int pixelsPerMeter, int margin) {
    cv::Mat court = draw(pixelsPerMeter, margin);
    const int courtH = static_cast<int>(HEIGHT * pixelsPerMeter);
    for (auto worldPoint : worldPoints) {
        const int px = margin + static_cast<int>(worldPoint.x * pixelsPerMeter);
        const int py = margin + courtH - static_cast<int>(worldPoint.y * pixelsPerMeter);
        const cv::Point center(px, py);
        cv::circle(court, center, 8, cv::Scalar(0, 255, 255), -1);
    }
    return court;
}

cv::Mat Court::drawVerification(const cv::Mat& frame) const {
    cv::Mat overlay = frame.clone();

    std::vector<cv::Point2f> projected;
    for (const auto& lm : landmarkWorldPositions()) {
        projected.push_back(worldToImage(lm));
    }

    for (int i = 0; i < projected.size(); i += 2) {
        cv::line(overlay, projected[i], projected[i + 1],
            cv::Scalar(0, 255, 0), 2);
    }

    for (int i = 0; i < projected.size() - 2; i += 2) {
        cv::line(overlay, projected[i], projected[i + 2],
            cv::Scalar(0, 255, 0), 2);
        cv::line(overlay, projected[i + 1], projected[i + 3],
            cv::Scalar(0, 255, 0), 2);
    }
    return overlay;
}

cv::Mat Court::warpToTopDown(const cv::Mat& frame, int pixelsPerMeter, int margin) const {
    cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
    S.at<double>(0, 0) = pixelsPerMeter;
    S.at<double>(1, 1) = -pixelsPerMeter;
    S.at<double>(0, 2) = margin;
    S.at<double>(1, 2) = 18*pixelsPerMeter + margin;
    cv::Mat warpH = S * homography_;

    int outW = 9 * pixelsPerMeter + 2 * margin;
    int outH = 18 * pixelsPerMeter + 2 * margin;

    cv::Mat topDown;
    cv::warpPerspective(frame, topDown, warpH,
        cv::Size(outW, outH));

    return topDown;
}

const cv::Mat& Court::getHomography() const {
    return homography_;
}

bool Court::isOnCourt(const cv::Point2f pixel, const float margin) const {
    const cv::Point2f world = imageToWorld(pixel);

    const bool onCourt = world.x > -margin && world.x < WIDTH + margin &&
        world.y > -margin && world.y < HEIGHT + margin;
    return onCourt;
}